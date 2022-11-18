from lux.kit import obs_to_game_state, GameState, EnvConfig, Team, Factory, Unit, UnitCargo
from lux.utils import direction_to
import numpy as np
import sys
from typing import Dict

class Agent():
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg
        self.a_net = ActionNet()
        self.a_net.load_state_dict(torch.load(f"action_net.pth"))

    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
        if step == 0:
            # bid 0 to not waste resources bidding and declare as the default faction
            return dict(faction="AlphaStrike", bid=0)
        else:
            game_state = obs_to_game_state(step, self.env_cfg, obs)
            # factory placement period

            # how much water and metal you have in your starting pool to give to new factories
            water_left = game_state.teams[self.player].water
            metal_left = game_state.teams[self.player].metal

            # how many factories you have left to place
            factories_to_place = game_state.teams[self.player].factories_to_place
            if factories_to_place > 0:
                # we will spawn our factory in a random location with 100 metal and water
                potential_spawns = game_state.board.spawns[self.player]
                spawn_loc = potential_spawns[np.random.randint(0, len(potential_spawns))]
                return dict(spawn=spawn_loc, metal=100, water=100)
            return dict()

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        state = obs_to_game_state(step, self.env_cfg, obs)
        state_tokens = env_to_tokens(state, self.player)
        a_tokens = self.a_net(torch.from_numpy(state_tokens))
        action_list = a_tokens.detatch().clone().numpy().tolist()
        actions = tokens_to_actions(state, action_list, self.player)
        return actions

from luxai2022 import LuxAI2022
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple
from torch.utils.data import DataLoader
import math
import random

#config
restart_epoch = 0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
token_len = 288
env_token_num = 10
action_token_num = 2

heads = 6
encoder_layers = 6
decoder_layers = 6
epochs = 2000
swing_range = 100
results_per_epoch = 3000
batch_size = 16

beta_max = 0.3
gamma_min = 0.5
gamma_max = 0.98
advantage_steps = 10

#盤面の評価に使える便利所たち
def resoure_exist(g_state:GameState, pos:np.ndarray, resource_type):#type = 1(ice) 0(ore)
        ice_existance = g_state.board.ice[pos[1]][pos[0]]
        ore_existance = g_state.board.ore[pos[1]][pos[0]]
        if resource_type == 0:
            return ice_existance == 1
        elif resource_type == 1:
            return ore_existance == 1
        else:
            print("resouce what")

def factory_adj_grids(factory:Factory):
    direction_list = np.array([[0,2], [1,2], [-1,2], [2,0], [2,-1], [2,1], [-2,1], [-2,0], [-2,-1], [0,-2], [1,-2], [-1,-2]])
    return [factory.pos + direction_list[i] for i in range(direction_list.shape[0])]

#便利な変換する奴らたち
def tokens_to_actions(state:GameState, tokens:np.ndarray, agent, unit_log):
    my_factories = state.factories[agent]
    my_units = state.units[agent]
    env_cfg:EnvConfig = state.env_cfg
 
    def factory_embedder(index):
        return_list = [0 for i in  range(token_len)]
        return_list[index + 1] = 1
        return return_list

    def robot_embedder(index):
        return_list = [0 for i in range(token_len)]
        return_list[0] = index // (token_len - 1) + 1
        return_list[-(index % (token_len - 1) + 1)] = 1
        return return_list

    def start_embedder(index):
        return_list = [0 for i in range(token_len)]
        if index == 0:
            return_list[-1] = 1
        elif index == 1:
            return_list[-2] = 1
        return return_list

    def pos_overrap_factory(g_state:GameState, pos:np.ndarray):
        factories = []
        for item in g_state.factories.values():
            factories.extend(list(item.values()))
        for factory in factories:
            factory_pos = factory.pos
            if abs(pos[0]-factory_pos[0]) < 3 and abs(pos[1]-factory_pos[1]) < 3:
                return True
        return False

    def unit_on_factory(g_state:GameState, unit:Unit):
        team_name = f"player_{unit.team_id}"
        factories = g_state.factories[team_name].values()
        for factory in factories:
            unit_pos = unit.pos
            factory_pos = factory.pos
            if abs(unit_pos[0]-factory_pos[0]) < 2 and abs(unit_pos[1]-factory_pos[1]) < 2:
                return True
        return False

    def unit_adjascent_factory(g_state:GameState, unit:Unit):
        team_name = f"player_{unit.team_id}"
        factories = g_state.factories[team_name].values()
        for factory in factories:
            unit_pos = unit.pos
            factory_pos = factory.pos
            if (abs(unit_pos[0]-factory_pos[0]) == 2 and abs(unit_pos[1]-factory_pos[1]) < 2)\
                or (abs(unit_pos[0]-factory_pos[0]) < 2 and abs(unit_pos[1]-factory_pos[1]) == 2):
                return True, factory
        return False, None

    def unit_next_action(unit:Unit):
        if(len(unit.action_queue) == 0):
            return [0,0,0,0,0]
        else:
            return unit.action_queue[0]

    def rulebased_factory(g_state:GameState, factory:Factory):
        action = None
        if factory.cargo.water - (env_cfg.max_episode_length - g_state.env_steps) >\
            factory.water_cost(g_state) * (env_cfg.max_episode_length - g_state.env_steps):
            print("rb watering")
            action = factory.water()
        return action

    def log_calc(g_state:GameState, unit:Unit, log:list):
        unit_cfg = unit.unit_cfg
        if len(log) < 2:
            print("error, not much log len")
        direction = direction_to(log[0], log[1])
        return_cost = 0
        return_cost += (len(log)-1) * unit_cfg.MOVE_COST
        rubble_map = g_state.board.rubble
        for i in range(1, len(log) - 1):
            pos = log[i]
            return_cost += rubble_map[pos[1]][pos[0]] * unit_cfg.RUBBLE_MOVEMENT_COST
        
        return unit.move(direction), return_cost

    def rulebased_unit(g_state:GameState, unit:Unit):
        action = None
        adj, factory = unit_adjascent_factory(g_state, unit)
        if unit_on_factory(g_state, unit) and unit.power < unit.dig_cost(g_state) * 3:
            #print(f"{unit.unit_id} rb pickup")
            action = unit.pickup(4, 50 if unit.unit_type == "LIGHT" else 100, True)
        elif adj:
            direction_factory = direction_to(unit.pos, factory.pos)
            if unit.power < unit.dig_cost(g_state) * 2 + unit.move_cost(g_state, direction_factory) and\
                unit.move_cost(g_state, direction_factory) + unit.action_queue_cost(g_state) < unit.power:
                action = unit.move(direction_factory)
                #print(f"{unit.unit_id} rb move dir = {direction_factory} from {unit.pos} to {factory.pos}")
            elif unit.power >= unit.dig_cost(g_state) + unit.move_cost(g_state, direction_factory):
                pos = unit.pos
                if resoure_exist(g_state, pos, 0) or resoure_exist(g_state, pos, 1):
                    print(f"{unit.unit_id} rb dig rubble = {g_state.board.rubble[pos[1]][pos[0]]} pos = {pos}, ice = {unit.cargo.ice}, ore = {unit.cargo.ore}")
                    print(f"ice = {g_state.board.ice[pos[1]][pos[0]]}, ore = {g_state.board.ore[pos[1]][pos[0]]}")
                    action = unit.dig(True)
            if unit.cargo.ice > unit.unit_cfg.DIG_RESOURCE_GAIN * 5 and\
                not all(unit_next_action(unit) == unit.move(direction_to(factory.pos, unit.pos))):
                action = unit.transfer(direction_factory, 0, unit.cargo.ice, False)
                #print(f"{unit.unit_id} rb passing ice {unit.cargo.ice}")
            if unit.cargo.ore > unit.unit_cfg.DIG_RESOURCE_GAIN * 5 and\
                not all(unit_next_action(unit) == unit.move(direction_to(factory.pos, unit.pos))):
                action = unit.transfer(direction_factory, 1, unit.cargo.ore, False)
                #print(f"{unit.unit_id} rb passing ore {unit.cargo.ore}")
        elif not unit_on_factory(g_state, unit):
            return_action, cost = log_calc(g_state, unit, unit_log[unit.unit_id])
            if unit.power < cost + unit.unit_cfg.MOVE_COST * 10 + unit.unit_cfg.DIG_COST and unit.power >= unit.unit_cfg.MOVE_COST:
                action = return_action
                #print(f"{unit.unit_id} remote return len {len(unit_log[unit.unit_id])-1} where {unit.pos}")
            elif unit.power >= unit.unit_cfg.DIG_COST and (resoure_exist(g_state, unit.pos, 0) or resoure_exist(g_state, unit.pos, 1)):
                action = unit.dig(True)
                #print(f"{unit.unit_id} remote dig pow {unit.power}")

        return action

    tokens = tokens.squeeze(0)

    actions = dict()
    if(state.real_env_steps >= 0):
        for index, factory in enumerate(my_factories.values()):
            action = rulebased_factory(state, factory)
            if not action == None:
                actions[factory.unit_id] = action
                continue
            embedder = factory_embedder(index)
            action_value = 0
            for i in range(token_len):
                action_value += embedder[i] * tokens[i]
            action_value = action_value * 3 % 3
            #print(f"favtory {factory.unit_id} action num = {action_value}")
            if action_value < 1:
                if factory.cargo.metal >= factory.build_heavy_metal_cost(state) and factory.power >= factory.build_heavy_power_cost(state):
                    actions[factory.unit_id] = factory.build_heavy()
            elif action_value < 2:
                if factory.cargo.metal >= factory.build_light_metal_cost(state) and factory.power >= factory.build_light_power_cost(state):
                    actions[factory.unit_id] = factory.build_light()
            elif action_value < 3:
                pass
            else:
                print("error-tipo")

        for index, unit in enumerate(my_units.values()):
            if(unit.power < unit.action_queue_cost(state)):
                continue
            
            action = rulebased_unit(state, unit)
            if action is None:
                embedder = robot_embedder(index)
                action_value = 0
                for i in range(token_len):
                    action_value += embedder[i] * tokens[i]
                action_value = action_value * 6 % 6
                #print(f"unit {unit.unit_id} action num = {action_value}")
                if  action_value < 1:
                    direction = int((action_value * 4) % 4)+1
                    cost = unit.move_cost(state, direction)
                    if not(cost == None) and unit.power >= cost:
                        action = unit.move(direction, True)
                elif action_value < 2:
                    #transferはrule_based
                    direction_dict = {1:[0,-1], 2:[1, 0], 3:[0,-1], 4:[-1,0]}
                    is_adj, factory = unit_adjascent_factory(state, unit)
                    if is_adj:
                        direction = direction_to(unit.pos, factory.pos)
                        resource_type = 0 if unit.cargo.ice > 0 else 1
                        amount = unit.cargo.ice if resource_type == 0 else unit.cargo.ore
                        action = unit.transfer(direction, resource_type, amount, False)
                        #if unit.cargo.ice>0 or unit.cargo.ore>0:
                            #print(f"{unit.unit_id} tarnser to {factory.unit_id}, {resource_type} {amount}")
                    else:
                        unit_list = []
                        for my_unit in my_units.values():
                            if abs(unit.pos[0]-my_unit.pos[0])+abs(unit.pos[1]-my_unit.pos[1])==1:
                                unit_list.append(my_unit)
                        if len(unit_list) == 0:
                            pass
                        else:
                            target_unit:Unit = unit_list[random.randint(0, len(unit_list)-1)]
                            direction = direction_to(unit.pos, target_unit.pos)
                            resource_type = 0 if unit.cargo.ice > 0 else 1
                            amount = unit.cargo.ice if resource_type == 0 else unit.cargo.ore
                            action = unit.transfer(direction, resource_type, amount, False)
                            #if unit.cargo.ice or unit.cargo.ore:
                                #print(f"{unit.unit_id} tarnser to {target_unit.unit_id}, {resource_type} {amount}")
                elif action_value < 3:
                    #pick_upはiceかoreのみ
                    if unit_on_factory(state, unit):
                        resource_type = int(((action_value-2) * 2) % 2)
                        action = unit.pickup(resource_type, 100, False)
                    elif unit.power >= unit.dig_cost(state):
                        action = unit.dig(False)
                elif action_value < 4:
                    if unit_on_factory(state, unit):
                        action = unit.pickup(4, unit.dig_cost(state), True)
                    elif unit.power >= unit.dig_cost(state):
                        action = unit.dig(True)
                elif action_value < 5:
                    #actions[unit.unit_id] = unit.self_destruct(False)
                    if unit_on_factory(state, unit):
                        action = unit.pickup(4, unit.dig_cost(state), True)
                    elif unit.power >= unit.dig_cost(state):
                        action = unit.dig(False)
                elif action_value < 6:
                    pass
                else:
                    print("error-tipo")
            
            if not (action is None) and not all(action == unit_next_action(unit)):
                actions[unit.unit_id] = [action]

    elif state.env_steps != 0:
        pos = np.zeros(2)
        for i in range(2):
            action_value = 0
            embedder = start_embedder(i)
            for k in range(token_len):
                action_value += embedder[k]*tokens[k]
            grid = math.ceil(action_value * 48 % 48)
            pos[i] = grid
        #print(f"set pos = {pos}")
        potential_spawns:np.ndarray = state.board.spawns[agent]
        length = 100
        index = 0
        for i in range(potential_spawns.shape[0]):
            if pos_overrap_factory(state, potential_spawns[i]):
                continue
            length_ = abs(potential_spawns[i][0]-pos[0])+abs(potential_spawns[i][1]-pos[1])
            if length_ < length:
                index = i
                length = length_
        actions = dict(spawn = potential_spawns[index], metal = 100, water = 100)
    else:
        actions = dict(faction="AlphaStrike", bid = 0)

    #print(f"actions = {actions}")
    return actions

def env_to_tokens(state:GameState, view_agent):#雑に作る。若干の情報のオーバーラップは仕方なし。
    board = state.board
    agents = []
    if view_agent == "player_0":
        agents = ["player_0", "player_1"]
    elif view_agent == "player_1":
        agents = ["player_1", "player_0"]
    else:
        print("error tipo")
    #まずはrubble
    rubble_map = board.rubble
    rubble_map = rubble_map.reshape((rubble_map.size // token_len, token_len))
    tokens = rubble_map
    #次にresources
    ice_map = board.ice
    ore_map = board.ore
    resource_map = ice_map + 2 * ore_map
    resource_map = resource_map.reshape((resource_map.size // token_len, token_len))
    tokens = np.concatenate((tokens, resource_map))
    #次にlichen
    lichen_map = board.lichen
    lichen_map = lichen_map.reshape((lichen_map.size // token_len, token_len))
    tokens = np.concatenate((tokens, lichen_map))
    #次にstrain
    strain_map = board.lichen_strains
    strain_map = strain_map.reshape((strain_map.size // token_len, token_len))
    tokens = np.concatenate((tokens, strain_map))
    #次にfactory(場所:2, リソース:5, strain_id:1, player:1)最高でも5個らしい
    factory_info = np.zeros((1, token_len))
    index_pos = 0
    for agent in agents:
        for factory in state.factories[agent].values():
            factory_info[0][index_pos] = factory.pos[0] / 48
            factory_info[0][index_pos+1] = factory.pos[1] / 48
            cargo:UnitCargo = factory.cargo
            factory_info[0][index_pos+2] = factory.power / 1000
            factory_info[0][index_pos+3] = cargo.ice / 100
            factory_info[0][index_pos+4] = cargo.ore / 100
            factory_info[0][index_pos+5] = cargo.water / 100
            factory_info[0][index_pos+6] = cargo.metal / 100
            factory_info[0][index_pos+7] = factory.strain_id / 100
            factory_info[0][index_pos+8] = 0 if agent == view_agent\
                else 1
            index_pos += 9
    tokens = np.concatenate((tokens, factory_info))
    #次にunit(場所:2, リソース:5, playerと種別:1, 次の行動:1)
    unit_info_dim = 9
    unit_num = 0
    for agent in state.units:
        unit_num += len(state.units[agent].values())
    #unit_infos = np.zeros((unit_num//(token_len//unit_info_dim) + 1, token_len))
    unit_infos = np.zeros((10, token_len))
    x_index = 0
    y_index = 0
    for agent in agents:
        for unit in state.units[agent].values():
            unit_infos[x_index][y_index] = unit.pos[0] / 48
            unit_infos[x_index][y_index+1] = unit.pos[1] / 48
            cargo:UnitCargo = unit.cargo
            unit_infos[x_index][y_index+2] = unit.power / 1000
            unit_infos[x_index][y_index+3] = cargo.ice / 100
            unit_infos[x_index][y_index+4] = cargo.ore / 100
            unit_infos[x_index][y_index+5] = cargo.water / 100
            unit_infos[x_index][y_index+6] = cargo.metal / 100
            if agent == view_agent:
                if unit.unit_type == "LIGHT":
                    unit_infos[x_index][y_index+7] = 0.5
                else:
                    unit_infos[x_index][y_index+7] = 1
            else:
                if unit.unit_type == "LIGHT":
                    unit_infos[x_index][y_index+7] = -0.5
                else:
                    unit_infos[x_index][y_index+7] = -1
            if len(unit.action_queue) > 0:
                next_action = unit.action_queue[0]
            else:
                next_action = [0, 0, 0, 0, 0]
            next_action_value = 125*next_action[0]+25*next_action[1]+5*next_action[2]+next_action[3]
            unit_infos[x_index][y_index+8] = next_action_value
            y_index += unit_info_dim
            if y_index+unit_info_dim > token_len:
                y_index = 0
                x_index += 1
    tokens = np.concatenate((tokens, unit_infos))
    #ラスト基本情報
    basics = np.zeros((1, token_len))
    basics[0][0] = state.real_env_steps / state.env_cfg.max_episode_length
    if state.env_cfg.max_episode_length - state.env_steps > token_len - 1:
        basics[0][1:-1] = state.weather_schedule[state.env_steps:state.env_steps+token_len-2]
    else:
        basics[0][1:1+state.env_cfg.max_episode_length - state.env_steps] = state.weather_schedule[state.env_steps:]
    tokens = np.concatenate((tokens, basics))
    
    return tokens

#近傍アクション生成機
def action_nearby_token(token:np.ndarray, variance):
    random_array = 3 * np.random.rand(*token.shape)
    token_ = (1 - variance) * token + variance * random_array
    #print(f"token = {token}")
    return token_

#stateの評価(技の見せ所)
def state_value(state:GameState, view_player):
    player = view_player
    opp_player = ""
    if player == "player_0":
        opp_player = "player_1"
    elif player == "player_1":
        opp_player = "player_0"
    else:
        print("player out of range")
    value = 0
    factories = state.factories
    my_factories = factories[player]
    opp_factories = factories[opp_player]
    #alive:O(e0)
    value += state.real_env_steps/100
    #print(f"val 0 = {value}")
    #lichens:O(e0)
    my_strains = []
    opp_strains = []
    for factory in my_factories.values():
        my_strains.append(factory.strain_id)
    for factory in opp_factories.values():
        opp_strains.append(factory.strain_id)
    board = state.board
    strain = board.lichen_strains
    lichen = board.lichen
    for i in range(strain.shape[0]):
        for k in range(strain.shape[1]):
            strain_id = strain[i][k]
            if strain_id in my_strains:
                value += lichen[i][k]/1000
            if strain_id in opp_strains:
                value -= lichen[i][k]/2000
    #print(f"val 1 = {value}")
    #factory_num:O(e-1)
    for factory in my_factories.values():
        value += min(factory.cargo.water/50, 1)/10
    for factory in opp_factories.values():
        value -= min(factory.cargo.water/50, 1)/20
    #print(f"val 2 = {value}")
    #//ここまで相手方の情報を考慮する
    my_units = state.units[player]
    #robot_num:O(e-1)
    for unit in my_units.values():
        if(unit.unit_type == "LIGHT"):
            value += 0
        elif(unit.unit_type == "HEAVY"):
            value += 0
    #print(f"val 3 = {value}")
    #factory_resources:O(e-2)
    for factory in my_factories.values():
        cargo = factory.cargo
        value += cargo.ice/2000 + cargo.water/1000
        if(factory.cargo.water > 100 - state.real_env_steps):
            value += 0.01
        adj_grids = factory_adj_grids(factory)
        for grid in adj_grids:
            if resoure_exist(state, grid, 0):
                value += 0.05
                #print(f"{factory.pos}, has ice in {grid}")
    #print(f"val 4 = {value}")
    #robot_resources:O(e-3)
    for unit in my_units.values():
        cargo = unit.cargo
        value += cargo.ice/4000
    #print(f"val 5 = {value}")
    
    return value

def action_value(state:GameState, action:Dict[str, np.ndarray], view_agent):
    value = 0

    my_units = state.units[view_agent]
    #資源の上を掘ったら報酬(e-2)
    for unit_id, unit in my_units.items():
        if unit.power > unit.dig_cost(state) * 2 and ((unit_id in action.keys() and (all(action[unit_id][0] == unit.dig(True)) or all(action[unit_id][0] == unit.dig(False)))) or\
            (not(unit_id in action.keys()) and len(unit.action_queue) > 0 and (all(unit.action_queue[0] == unit.dig(True)) or all(unit.action_queue[0] == unit.dig(False))))):
            pos = unit.pos
            ice_map = state.board.ice
            ore_map = state.board.ore
            if ice_map[pos[1]][pos[0]] > 0:
                value += 0.01
                #print("digging ice")
            elif ore_map[pos[1]][pos[0]] > 0:
                value += 0.005
                #print("digging ore")

    return value

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x:torch.Tensor):
        if x.dim() == 3:
            pe = self.pe
            x = x + pe[:x.size(0), :].to(device)
        else:
            pe = self.pe.squeeze(1)
            x = x + pe[:x.size(0), :].to(device)
        return self.dropout(x)

class ActionNet(nn.Module):#stateからactionを導く。学習度判定にも使用します。
    def __init__(self):
        super(ActionNet, self).__init__()
        self.pos_encoder = PositionalEncoding(token_len)
        self.transformer = nn.Transformer(d_model=token_len, dim_feedforward=token_len*2)
        self.sequence = nn.Sequential(\
            nn.Linear(token_len, token_len),\
            nn.ReLU(),\
            nn.Linear(token_len, token_len))

    def forward(self, state):
        x:torch.Tensor = self.pos_encoder(state)
        decoder_input = torch.from_numpy(np.ones([1, x.shape[1], x.shape[2]]) if x.dim() == 3 else\
            np.ones([1, x.shape[1]])).float().to(device)
        x = self.transformer(x, decoder_input)
        out = self.sequence(x)
        return out

class ValueNet(nn.Module):#stateとactionからvalueを導く。
    def __init__(self):
        super(ValueNet, self).__init__()
        self.pos_encoder = PositionalEncoding(token_len)
        self.transformer = nn.Transformer(d_model=token_len, dim_feedforward=token_len*2)
        self.sequence = nn.Sequential(\
            nn.Linear(token_len, token_len//2),\
            nn.ReLU(),\
            nn.Linear(token_len//2, 1))

    def forward(self, state):
        x = self.pos_encoder(state)
        decoder_input = torch.from_numpy(np.ones([1, x.shape[1], x.shape[2]]) if x.dim() == 3 else\
            np.ones([1, x.shape[1]])).float().to(device)
        x = self.transformer(x, decoder_input)
        out = self.sequence(x)
        return out

class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        self.pos_encoder = PositionalEncoding(token_len)
        self.transformer = nn.Transformer(d_model=token_len, dim_feedforward=token_len*2)
        self.sequence = nn.Sequential(\
            nn.Linear(token_len, token_len),\
            nn.ReLU(),\
            nn.Linear(token_len, token_len//2))

    def forward(self, state):
        x = self.pos_encoder(state)
        decoder_input = torch.from_numpy(np.ones([1, x.shape[1], x.shape[2]]) if x.dim() == 3 else\
            np.ones([1, x.shape[1]])).float().to(device)
        x = self.transformer(x, decoder_input)
        out = self.sequence(x)
        return out

class DataSet(torch.utils.data.Dataset):
    def __init__(self) -> None:
        self.s = []
        self.a = []
        self.v = []
        super().__init__()

    def add_items(self, _s, _a, _v):
        self.s.append(_s)
        self.a.append(_a)
        self.v.append(_v)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out_s = self.s[idx]
        out_a = self.a[idx]
        out_v = self.v[idx]
        out_s = torch.from_numpy(out_s).float()
        out_a = torch.from_numpy(out_a).float()
        out_v = torch.from_numpy(out_v).float()
        out_s.to(device)
        out_a.to(device)
        out_v.to(device)
        return out_s, out_a, out_v

    def __len__(self) -> int:
        return len(self.s)

def Play(v_net: ValueNet, a_net: ActionNet, d_net:CustomNet, s_net:CustomNet, beta):
    def unit_on_factory(state:GameState, unit:Unit):
        team_name = f"player_{unit.team_id}"
        factories = state.factories[team_name].values()
        for factory in factories:
            unit_pos = unit.pos
            factory_pos = factory.pos
            if abs(unit_pos[0]-factory_pos[0]) < 2 and abs(unit_pos[1]-factory_pos[1]) < 2:
                return True
        return False
   
    def log_addition(log:list, pos:np.ndarray):
        def check_is_inlog():
            exist = False
            index = -1
            for i in range(len(log)):
                if (pos == log[i]).all():
                    exist = True
                    index = i
                    break
            return exist, index

        exist, index = check_is_inlog()
        if exist:
            return log[index:]
        else:
            log.insert(0, pos)
            return log

    def check_is_finished(state:GameState, game_len):
        finished = False
        cause:str = ""
        factories = state.factories.items()
        for agent, factory in factories:
            if len(factory) == 0 and state.real_env_steps > 0:
                finished = True
                cause += f"[{agent} down] "
        if state.env_steps == game_len - 1:
            finished = True
            cause += f"[env leached last]"
        return finished, cause
    env = LuxAI2022(verbose = 0)
    seed = random.randint(0, 10000)
    step = 0
    env_cfg = env.env_cfg
    obs = env.reset(seed = seed)["player_0"]
    state = obs_to_game_state(step, env_cfg, obs)
    unit_log_0 = {}
    unit_log_1 = {}
    state_tokens_0 = env_to_tokens(state, "player_0")
    state_tokens_1 = env_to_tokens(state, "player_1")
    state_value_0 = state_value(state, "player_0")
    state_value_1 = state_value(state, "player_1")
    states_0 = [state_tokens_0]
    actions_0 = []
    returns_0 = []
    states_1 = [state_tokens_1]
    actions_1 = []
    returns_1 = []
    while not check_is_finished(state, env_cfg.max_episode_length)[0]:#環境が終わるまで
        #まずはlogの更新から
        units_0 = state.units["player_0"]
        for unit in units_0.values():
            if not(unit.unit_id in unit_log_0.keys()) or unit_on_factory(state, unit):
                unit_log_0[unit.unit_id] = [unit.pos]
            else:
                unit_log_0[unit.unit_id] = log_addition(unit_log_0[unit.unit_id], unit.pos)
        units_1 = state.units["player_1"]
        for unit in units_1.values():
            if not(unit.unit_id in unit_log_1.keys()) or unit_on_factory(state, unit):
                unit_log_1[unit.unit_id] = [unit.pos]
            else:
                unit_log_1[unit.unit_id] = log_addition(unit_log_1[unit.unit_id], unit.pos)

        state_tokens_0 = torch.from_numpy(env_to_tokens(state, "player_0")).float().to(device)
        state_tokens_1 = torch.from_numpy(env_to_tokens(state, "player_1")).float().to(device)
        a_0_t = a_net(state_tokens_0)
        a_1_t = a_net(state_tokens_1)
        a_0 = a_0_t.to('cpu').detach().numpy().copy()
        a_1 = a_1_t.to('cpu').detach().numpy().copy()
        value_0 = v_net(torch.concat([state_tokens_0, a_0_t])).item()
        default_0:np.ndarray = d_net(torch.concat([state_tokens_0, a_0_t]).unsqueeze(0)).to('cpu').detach().numpy().copy()
        search_0:np.ndarray = s_net(torch.concat([state_tokens_0, a_0_t]).unsqueeze(0)).to('cpu').detach().numpy().copy()
        search_mse_0 = ((default_0 - search_0) ** 2).mean()
        value_1 = v_net(torch.concat([state_tokens_1, a_1_t])).item()
        default_1:np.ndarray = d_net(torch.concat([state_tokens_1, a_1_t]).unsqueeze(0)).to('cpu').detach().numpy().copy()
        search_1:np.ndarray = s_net(torch.concat([state_tokens_1, a_1_t]).unsqueeze(0)).to('cpu').detach().numpy().copy()
        search_mse_1 = ((default_1 - search_1) ** 2).mean()
        for i in range(10):#もちろん時間で区切ってもよし
            a_0_ = action_nearby_token(a_0, beta)
            a_0_t_ = torch.from_numpy(a_0_).float().to(device)
            value_0_ = v_net(torch.concat([state_tokens_0, a_0_t_])).item()
            default_0_:np.ndarray = d_net(torch.concat([state_tokens_0, a_0_t_]).unsqueeze(0)).to('cpu').detach().numpy().copy()
            search_0_:np.ndarray = s_net(torch.concat([state_tokens_0, a_0_t_]).unsqueeze(0)).to('cpu').detach().numpy().copy()
            search_mse_0_ = ((default_0_ - search_0_) ** 2).mean()
            a_1_ = action_nearby_token(a_1, beta)
            a_1_t_ = torch.from_numpy(a_1_).float().to(device)
            value_1_ = v_net(torch.concat([state_tokens_1, a_1_t_])).item()
            default_1_:np.ndarray = d_net(torch.concat([state_tokens_1, a_1_t_]).unsqueeze(0)).to('cpu').detach().numpy().copy()
            search_1_:np.ndarray = s_net(torch.concat([state_tokens_1, a_1_t_]).unsqueeze(0)).to('cpu').detach().numpy().copy()
            search_mse_1_ = ((default_1_ - search_1_) ** 2).mean()
            #print("val = {0:3g}, beta = {1:3g}".format((value_0_ - value_0)/(abs(value_0)+1e-10), (search_mse_0_ - search_mse_0)/search_mse_0 * beta * 50))
            if ((value_0_ - value_0)/(abs(value_0)+1e-10) * (1-beta) + (search_mse_0_ - search_mse_0)/search_mse_0 * beta) > 0:
                a_0 = a_0_
                value_0 = value_0_
                search_mse_0 = search_mse_0_
            if ((value_1_ - value_1)/(abs(value_1)+1e-10) * (1-beta) + (search_mse_1_ - search_mse_1)/search_mse_1 * beta / 10) > 0:
                a_1 = a_1_
                value_1 = value_1_
                search_mse_1 = search_mse_1_
        actions_0.append(a_0)
        actions_1.append(a_1)
        real_actions_0 = tokens_to_actions(state, a_0, "player_0", unit_log_0)
        action_value_0 = action_value(state, real_actions_0, "player_0")
        real_actions_1 = tokens_to_actions(state, a_1, "player_1", unit_log_1)
        action_value_1 = action_value(state, real_actions_1, "player_1")
        real_actions = {"player_0":real_actions_0, "player_1":real_actions_1}
        obs = env.step(real_actions)[0]["player_0"]
        step += 1
        state = obs_to_game_state(step, env_cfg, obs)
        state_tokens_0 = env_to_tokens(state, "player_0")
        state_tokens_1 = env_to_tokens(state, "player_1")
        states_0.append(state_tokens_0)
        states_1.append(state_tokens_1)
        state_value_0_ = state_value(state, "player_0")
        returns_0.append(state_value_0_ - state_value_0 + action_value_0)
        state_value_0 = state_value_0_
        state_value_1_ = state_value(state, "player_1")
        returns_1.append(state_value_1_ - state_value_1 + action_value_1)
        state_value_1 = state_value_1_
        #print(f"0 return = {returns_0[-1]} value = {state_value_0}")
        #print(f"1 return = {returns_1[-1]} value = {state_value_1}")             
    print(f"env {seed} finished in step {state.real_env_steps} cause = {check_is_finished(state, env_cfg.max_episode_length)[1]}")

    return (states_0, actions_0, returns_0),(states_1, actions_1, returns_1)#各ステップのenv, act, returnの配列を吐き出させる

def Play_with_display(value_net: ValueNet, action_net: ActionNet, variance):
    while False:#環境が終わるまで
        environment = 0
        a = action_net(environment)
        value = value_net(torch.concat([environment, a]))
        for i in range(10):#もちろん時間で区切ってもよし
            a_ = action_nearby_token(a)
            value_ = value_net(environment, a_)
            if(value_ > value):
                a = a_
                value = value_
    return [], [], [] #各ステップのenv, act, returnの配列を吐き出させる

def Update(results, a_net:ActionNet, v_net:ValueNet, d_net:CustomNet, s_net:CustomNet, gamma):#マッチの結果をもとに学習をする。
    train_set = DataSet()
    for result in results:
        steps = len(result[1])
        #print(result[2])
        for i in range(steps):
            s, s_, a = result[0][i], result[0][i+1], result[1][i]
            v = 0
            for j in range(i, min(steps, i + advantage_steps)):
                v += result[2][j] * pow(gamma, j-i)
            if (steps > i+advantage_steps):
                last_state_tensor = torch.from_numpy(result[0][i + advantage_steps + 1]).float().to(device)
                last_value:torch.Tensor = v_net(last_state_tensor)
                v += last_value.detach().item() * pow(gamma, advantage_steps)
            #print(f"step {i} value is {v}")
            v = np.array([v])
            train_set.add_items(s, a, v)
    print(f"made train set")
    
    square_loss = nn.MSELoss()
    a_optimiser = optim.SGD(a_net.parameters(), lr=5e-4, momentum=0.9, nesterov= True)
    v_optimizer = optim.SGD(v_net.parameters(), lr=5e-4, momentum=0.9, nesterov= True)
    s_optimizer = optim.SGD(s_net.parameters(), lr=5e-4, momentum=0.9, nesterov= True)
    running_loss_a = 0.0
    running_loss_v = 0.0
    running_loss_s = 0.0
    v_total = 0.0
    total_batch = 0
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    for j, (s, a, v) in enumerate(train_loader, 0):
        total_batch += 1
        a_optimiser.zero_grad()
        v_optimizer.zero_grad()
        s_optimizer.zero_grad()

        s = torch.transpose(s, 0, 1).to(device)#[N, B, Length]
        a = torch.transpose(a, 0, 1).to(device)#[1, B, Length]
        v = torch.unsqueeze(v, 0).to(device)#[1, N, 1]
        #print(f"tgt v = {v}")
        v_total += torch.mean(v).item()

        a_output = a_net(s)
        a_loss = square_loss(a_output, a)
        a_loss.backward()
        a_optimiser.step()
        running_loss_a += a_loss.item()
        v_output = v_net(torch.concat((s, a)))
        #print(f"src v = {v_output}")
        v_loss:torch.Tensor = square_loss(v_output, v)
        v_loss.backward()
        v_optimizer.step()
        running_loss_v += v_loss.item()

        d_output:torch.Tensor = d_net(s)
        d_output = d_output.detach()
        s_output = s_net(s)
        s_loss = square_loss(s_output, d_output)
        s_loss.backward()
        s_optimizer.step()
        running_loss_s += s_loss.item()
    print("running loss [v: {0:.3g}, a: {1:.3g}, s: {2:.3g}]"\
        .format(running_loss_v/total_batch, running_loss_a/total_batch, running_loss_s/total_batch))
    print("value ave {0:3g}".format(v_total/total_batch))
    running_loss_a = 0.0
    running_loss_v = 0.0
    running_loss_s = 0.0
    v_total = 0.0
    total_batch = 0
    print("updated")

def Train():
    action_net = ActionNet().to(device)
    value_net = ValueNet().to(device)
    default_net = CustomNet().to(device)
    search_net = CustomNet().to(device)
    if(restart_epoch > 0):
        action_net.load_state_dict(torch.load(f"model_a_{restart_epoch-1}.pth", map_location= device))
        value_net.load_state_dict(torch.load(f"model_v_{restart_epoch-1}.pth", map_location= device))
        default_net.load_state_dict(torch.load(f"model_d_{restart_epoch-1}.pth", map_location= device))
        search_net.load_state_dict(torch.load(f"model_s_{restart_epoch-1}.pth", map_location= device))

    for i in range(epochs - restart_epoch):
        #Playから
        action_net.eval()
        value_net.eval()
        default_net.eval()
        search_net.eval()
        variance = abs(1-(((i+restart_epoch)%(swing_range*2)))/swing_range) * (1 - (i + restart_epoch)/epochs)
        results = []
        results_num = 0
        while results_num < results_per_epoch:
            result_1, result_2 = Play(value_net, action_net, default_net, search_net,\
                (beta_max * variance))
            results.append(result_1)
            results_num += len(result_1[1])
            results.append(result_2)
            results_num += len(result_2[1])
        #ネットワークアップデート
        action_net.train()
        value_net.train()
        default_net.train()
        search_net.train()
        print("average episode len {0:3g},variance = {3:3g}, gamma = {1:3g}, beta = {2:3g}".format(results_num/len(results), gamma_max, beta_max * variance, variance))
        Update(results, action_net, value_net, default_net, search_net, gamma_max)
        
        if (i + restart_epoch)%10 == 0:
            torch.save(action_net.state_dict(), f"model_a_{restart_epoch + i}.pth")
            torch.save(value_net.state_dict(), f"model_v_{restart_epoch + i}.pth")
            torch.save(default_net.state_dict(), f"model_d_{restart_epoch + i}.pth")
            torch.save(search_net.state_dict(), f"model_s_{restart_epoch + i}.pth")
        print(f"epoch {i + restart_epoch}")

if __name__ == "__main__":
    arg = sys.argv
    if arg[1] == "__train":
        #訓練の挙動を定義
        print(f"ver1.7.4 restart from epoch {restart_epoch}")
        Train()
    elif arg[1] == "__predict":
        #実行の挙動を定義
        pass