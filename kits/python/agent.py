#こっちは提出の時に参照される方
from lux.kit import obs_to_game_state, GameState, EnvConfig
from lux.utils import direction_to
import numpy as np
import sys
import torch
import torch.nn as nn
from lux.kit import Team, Factory, Unit, UnitCargo
import math
import os

model_path = os.path.dirname(__file__) + "/models/a_model.pth"
print("from agent.py model path isfile = {0}".format(os.path.isfile(model_path)))

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
beta_max = 0.5
batch_size = 16

gamma_max = 0.98
advantage_steps = 3

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

#便利な変換する奴らたち
def tokens_to_actions(state:GameState, tokens:np.ndarray, agent):
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

    def pos_overrap_factory(state:GameState, pos:np.ndarray):
        factories = []
        for item in state.factories.values():
            factories.extend(list(item.values()))
        for factory in factories:
            factory_pos = factory.pos
            if abs(pos[0]-factory_pos[0]) < 3 and abs(pos[1]-factory_pos[1]) < 3:
                return True
        return False

    def unit_on_factory(state:GameState, unit:Unit):
        team_name = f"player_{unit.team_id}"
        factories = state.factories[team_name].values()
        for factory in factories:
            unit_pos = unit.pos
            factory_pos = factory.pos
            if abs(unit_pos[0]-factory_pos[0]) < 2 and abs(unit_pos[1]-factory_pos[1]) < 2:
                return True
        return False

    def unit_adjascent_factory(state:GameState, unit:Unit):
        team_name = f"player_{unit.team_id}"
        factories = state.factories[team_name].values()
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

    def rulebased_factory(state:GameState, factory:Factory):
        action = None
        if factory.cargo.water - (env_cfg.max_episode_length - state.env_steps) >\
            factory.water_cost(state) * (env_cfg.max_episode_length - state.env_steps):
            print("rb watering")
            action = factory.water()
        return action

    def rulebased_unit(state:GameState, unit:Unit):
        action = None
        if unit_on_factory(state, unit) and unit.power < unit.dig_cost(state) * 3:
            #print(f"rb pickup id = {unit.unit_id}")
            action = unit.pickup(4, 50 if unit.unit_type == "LIGHT" else 100, True)
        adj, factory = unit_adjascent_factory(state, unit)
        if adj:
            direction_factory = direction_to(unit.pos, factory.pos)
            if unit.power < unit.dig_cost(state) + unit.move_cost(state, direction_factory) and\
                unit.move_cost(state, direction_factory) + unit.action_queue_cost(state) < unit.power:
                action = unit.move(direction_factory)
                #print(f"rb move dir = {direction_factory} from {unit.pos} to {factory.pos}, id = {unit.unit_id}")
            else:
                pos = unit.pos
                ice_existance = state.board.ice[pos[1]][pos[0]]
                ore_existance = state.board.ore[pos[1]][pos[0]]
                if state.board.ice[pos[1]][pos[0]] == 1 or state.board.ore[pos[1]][pos[0]] == 1:
                    #print(f"rb dig rubble = {state.board.rubble[pos[1]][pos[0]]} pos = {pos}, ice = {unit.cargo.ice}, ore = {unit.cargo.ore}")
                    #print(f"ice = {state.board.ice[pos[1]][pos[0]]}, ore = {state.board.ore[pos[1]][pos[0]]}, id = {unit.unit_id}")
                    action = unit.dig(True)
            if unit.cargo.ice > unit.unit_cfg.DIG_RESOURCE_GAIN * 5 and\
                not all(unit_next_action(unit) == unit.move(direction_to(factory.pos, unit.pos))):
                action = unit.transfer(direction_factory, 0, unit.cargo.ice, False)
                #print(f"rb passing ice {unit.cargo.ice}, id = {unit.unit_id}")
            if unit.cargo.ore > unit.unit_cfg.DIG_RESOURCE_GAIN * 5 and\
                not all(unit_next_action(unit) == unit.move(direction_to(factory.pos, unit.pos))):
                action = unit.transfer(direction_factory, 1, unit.cargo.ore, False)
                #print(f"rb passing ore {unit.cargo.ore}, id = {unit.unit_id}")
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
            action_value = action_value % 3
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
                action_value = action_value % 6
                if  action_value < 1:
                    direction = int(((action_value - 1)*4) % 4)+1
                    cost = unit.move_cost(state, direction)
                    if not(cost == None) and unit.power >= cost:
                        action = unit.move(direction, True)
                elif action_value < 2:
                    #transferはiceかoreのみ
                    resource_type = int(((action_value-1)*3) % 3)
                    direction = int(((action_value - 1)*20) % 4)+1
                    direction_vec = [((direction+1)//2)*(3-direction), ((direction)//2)*(direction-2)]
                    destination = [unit.pos[0] + direction_vec[0], unit.pos[1] + direction_vec[1]]
                    if -1 < destination[0] and destination[0] < env_cfg.map_size and -1 < destination[1] and destination[1] < env_cfg.map_size:
                        if resource_type == 0:
                            action = unit.transfer(direction, 0, unit.cargo.ice, False)
                        elif resource_type == 1:
                            action = unit.transfer(direction, 1, unit.cargo.ore, False)
                        elif resource_type == 2:
                            action = unit.transfer(direction, 4, unit.power//2, False)
                        else:
                            print("どのリソースをtransferするん？")
                elif action_value < 3:
                    #pick_upはiceかoreのみ
                    if unit_on_factory(state, unit):
                        resource_type = int(((action_value-2)*2) % 2)
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
        print(f"{agent} spawn at {pos}")
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
    random_array = 2 * variance * np.random.rand(*token.shape) - variance
    token_ = token + random_array
    return token_

def state_value(state:GameState, view_player):
    player = view_player
    opp_player = "player_1" if player == "player_0" else "player_0"
    value = 0
    factories = state.factories
    my_factories = factories[player]
    opp_factories = factories[opp_player]
    #alive:O(e0)
    value += state.real_env_steps/1000
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
                value -= lichen[i][k]/1000
    #print(f"val 1 = {value}")
    #factory_num:O(e-1)
    for factory in my_factories.values():
        value += min(factory.cargo.water/50, 1)
    for factory in opp_factories.values():
        value -= min(factory.cargo.water/50, 1)
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
    #print(f"val 4 = {value}")
    #robot_resources:O(e-3)
    for unit in my_units.values():
        cargo = unit.cargo
        value += cargo.ice/4000
    #print(f"val 5 = {value}")
    
    return value

class Agent():
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg
        self.a_net = ActionNet()
        self.a_net.to(device)
        self.a_net.load_state_dict(torch.load(model_path, map_location= device))
        self.post_state_value = 0

    def determin_action(self, step:int, obs, remainingOverageTime: int = 60):
        state = obs_to_game_state(step, self.env_cfg, obs)
        tokens = torch.from_numpy(env_to_tokens(state, self.player)).float().to(device)
        a_t = self.a_net(tokens)
        a = a_t.to('cpu').detach().numpy().copy()
        real_action = tokens_to_actions(state, a, self.player)
        return real_action

    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
        return self.determin_action(step, obs, remainingOverageTime)

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        state = obs_to_game_state(step, self.env_cfg, obs)
        print("statevalue step {0} is {1:3g}, {2:3g}".format(state.real_env_steps, state_value(state, "player_0"), state_value(state, "player_1")))
        return self.determin_action(step, obs, remainingOverageTime)