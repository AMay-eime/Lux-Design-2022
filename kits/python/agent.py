#こっちは提出の時に参照される方
from lux.kit import obs_to_game_state, GameState, EnvConfig
from lux.utils import direction_to, is_the_same_action
import numpy as np
import sys
import torch
import torch.nn as nn
from lux.kit import Team, Factory, Unit, UnitCargo
import math
import os
from typing import Tuple
import random

a_model_path = os.path.dirname(__file__) + "/models/a_model.pth"
v_model_path = os.path.dirname(__file__) + "/models/v_model.pth"

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

#rule basedを制御する変数
target_light_num = 3
factory_territory = 2
least_water_storage = 400

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

def rubble_num(g_state:GameState, pos:np.ndarray):
    return g_state.board.rubble[pos[1]][pos[0]]

def factory_adj_grids(factory:Factory, env_cfg:EnvConfig):
    direction_list = [[0,2], [1,2], [-1,2], [2,0], [2,-1], [2,1], [-2,1], [-2,0], [-2,-1], [0,-2], [1,-2], [-1,-2]]
    return_list = []
    for i in range(len(direction_list)):
        pos = factory.pos + np.array(direction_list[i])
        if -1 < pos[0] and pos[0] < env_cfg.map_size and -1 < pos[1] and pos[1] < env_cfg.map_size:
            return_list.append(pos)
    return return_list

def unit_on_factory(g_state:GameState, unit:Unit):
    team_name = f"player_{unit.team_id}"
    factories = g_state.factories[team_name].values()
    for factory in factories:
        unit_pos = unit.pos
        factory_pos = factory.pos
        if abs(unit_pos[0]-factory_pos[0]) < 2 and abs(unit_pos[1]-factory_pos[1]) < 2:
            return True, factory
    return False, None

def pos_on_factory(g_state:GameState, pos:np.ndarray):
    factories = list(g_state.factories["player_1"].values())
    factories.extend(list(g_state.factories["player_0"].values()))
    for factory in factories:
        factory_pos = factory.pos
        if abs(pos[0]-factory_pos[0]) < 2 and abs(pos[1]-factory_pos[1]) < 2:
            return True, factory
    return False, None

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
            if abs(pos[0]-factory_pos[0]) < 3 + factory_territory*2 and abs(pos[1]-factory_pos[1]) < 3 + factory_territory*2:
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
        if factory.cargo.water - (env_cfg.max_episode_length - g_state.real_env_steps) >\
            factory.water_cost(g_state) * (env_cfg.max_episode_length - g_state.real_env_steps):
            #print("rb watering")
            action = factory.water()
        elif factory.cargo.metal >= factory.build_heavy_metal_cost(state) and factory.power >= factory.build_heavy_power_cost(state):
            action = factory.build_heavy()
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
        is_on, factory_on = unit_on_factory(g_state, unit)
        exist, factory_base = pos_on_factory(g_state, unit_log[unit.unit_id][0][-1])

        if unit.unit_type == "LIGHT":
            search_list = np.array([[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,-1],[-1,1],[1,-1],[2,0],[-2,0],[0,2],[0,-2]])
            heavy_list = []
            target_pos = None
            for unit_ in state.units["player_1"].values():
                if unit_.unit_type == "HEAVY":
                    heavy_list.append(unit_.pos)
            for unit_ in state.units["player_0"].values():
                if unit_.unit_type == "HEAVY":
                    heavy_list.append(unit_.pos)
            for i in range(search_list.shape[0]):
                pos = unit.pos + search_list[i]
                for pos_ in heavy_list:
                    if all(pos == pos_):
                        target_pos = pos
                        break
                if not(target_pos is None):
                    break
            if not(target_pos is None):
                action = unit.move(direction_to(target_pos, unit.pos))

        if unit.unit_type == "HEAVY":
            #隣に敵のHEAVYがいる場合は突進する(この判定は独自に行う（優先度最低）
            enemy_team_id = "player_1" if unit.team_id == 0 else "player_0"
            enemy_units = g_state.units[enemy_team_id].values()
            target_unit = None
            for enemy_unit in enemy_units:
                ds = [enemy_unit.pos[0] - unit.pos[0], enemy_unit.pos[1] - unit.pos[1]]
                dist = abs(ds[0]) + abs(ds[1])
                if dist == 1 and enemy_unit.unit_type == "HEAVY":
                    target_unit = enemy_unit
            if not(target_unit is None) and not pos_on_factory(g_state, target_unit.pos)[0]:
                print(f"in {g_state.real_env_steps}, {unit.unit_id} charges {target_unit.unit_id}!")
                action = unit.move(direction_to(unit.pos, target_unit.pos))
        if unit.unit_type == "HEAVY" and exist and (factory_base.cargo.water < least_water_storage or env_cfg.max_episode_length * 0.8 < g_state.real_env_steps):
            #初期生産でかつ水資源に余裕がなければ所属ファクトリー周辺の水資源を探す(生存本能でないから優先度は低い)
            #さらに、最後の方では水やりをするためにたっぷり水を確保してくる
            search_center = factory_base.pos
            adj_vecs = np.array([[2,0],[2,1],[2,-1],[-2,0],[-2,1],[-2,-1],[0,2],[1,2],[-1,2],[0,-2],[1,-2],[-1,-2]])
            second_adj_vecs = np.array([[3,0],[3,1],[3,-1],[-3,0],[-3,1],[-3,-1],[0,3],[1,3],[-1,3],[0,-3],[1,-3],[-1,-3],[2,2],[2,-2],[-2,2],[-2,-2]])
            ice_pos = None
            for i in range(adj_vecs.shape[0]):
                target_pos = search_center + adj_vecs[i]
                if target_pos[0] < 0 or env_cfg.map_size-1 < target_pos[0] or target_pos[1] < 0 or env_cfg.map_size-1 < target_pos[1]:
                    continue
                if resoure_exist(g_state, search_center + adj_vecs[i], 0):
                    ice_pos = search_center+adj_vecs[i]
                    break
            if ice_pos is None:
                for i in range(second_adj_vecs.shape[0]):
                    target_pos = search_center + second_adj_vecs[i]
                    if target_pos[0] < 0 or env_cfg.map_size-1 < target_pos[0] or target_pos[1] < 0 or env_cfg.map_size-1 < target_pos[1]:
                        continue
                    if resoure_exist(g_state, search_center + second_adj_vecs[i], 0):
                        ice_pos = search_center+second_adj_vecs[i]
                        break
            if not (ice_pos is None):
                #print(f"{unit.unit_id} is assigned {factory_base.unit_id} found ice at {ice_pos}")
                action = unit.move(direction_to(unit.pos, ice_pos))
        elif unit.unit_type == "HEAVY" and exist:
            #場に自分のlight_unitが一定数以下しか存在しない場合は自身の周りにoreがある場合に掘りに行く(優先度さらに低)
            total_light_num = 0
            for unit_ in my_units.values():
                if unit_.unit_type == "LIGHT":
                    total_light_num += 1
            if total_light_num < target_light_num:
                search_center = factory_base.pos
                adj_vecs = np.array([[2,0],[2,1],[2,-1],[-2,0],[-2,1],[-2,-1],[0,2],[1,2],[-1,2],[0,-2],[1,-2],[-1,-2]])
                second_adj_vecs = np.array([[3,0],[3,1],[3,-1],[-3,0],[-3,1],[-3,-1],[0,3],[1,3],[-1,3],[0,-3],[1,-3],[-1,-3],[2,2],[2,-2],[-2,2],[-2,-2]])
                ore_pos = None
                for i in range(adj_vecs.shape[0]):
                    target_pos = search_center + adj_vecs[i]
                    if target_pos[0] < 0 or env_cfg.map_size-1 < target_pos[0] or target_pos[1] < 0 or env_cfg.map_size-1 < target_pos[1]:
                        continue
                    if resoure_exist(g_state, search_center + adj_vecs[i], 1):
                        ore_pos = search_center+adj_vecs[i]
                        break
                if ore_pos is None:
                    for i in range(second_adj_vecs.shape[0]):
                        target_pos = search_center + second_adj_vecs[i]
                        if target_pos[0] < 0 or env_cfg.map_size-1 < target_pos[0] or target_pos[1] < 0 or env_cfg.map_size-1 < target_pos[1]:
                            continue
                        if resoure_exist(g_state, search_center + second_adj_vecs[i], 1):
                            ore_pos = search_center+second_adj_vecs[i]
                            break
                if not (ore_pos is None):
                    action = unit.move(direction_to(unit.pos, ore_pos))

        if is_on and factory_on.power > 100 and unit.power < unit.unit_cfg.DIG_COST * 3:
            #print(f"{unit.unit_id} rb pickup")
            action = unit.pickup(4, min(unit.unit_cfg.DIG_COST * 5, factory_on.power), True)
        elif adj:
            direction_factory = direction_to(unit.pos, factory.pos)
            if unit.power < unit.dig_cost(g_state) * 2 + unit.move_cost(g_state, direction_factory):
                action = unit.move(direction_factory)
                #print(f"{unit.unit_id} rb move dir = {direction_factory} from {unit.pos} to {factory.pos}")
            elif unit.power >= unit.dig_cost(g_state) + unit.move_cost(g_state, direction_factory):
                pos = unit.pos
                if resoure_exist(g_state, pos, 0) or resoure_exist(g_state, pos, 1):
                    #print(f"{unit.unit_id} rb dig rubble = {g_state.board.rubble[pos[1]][pos[0]]} pos = {pos}, ice = {unit.cargo.ice}, ore = {unit.cargo.ore}")
                    #print(f"ice = {g_state.board.ice[pos[1]][pos[0]]}, ore = {g_state.board.ore[pos[1]][pos[0]]}")
                    action = unit.dig(True)
            if unit.cargo.ice > unit.unit_cfg.DIG_RESOURCE_GAIN * 5 and\
                not all(unit_next_action(unit) == unit.move(direction_to(factory.pos, unit.pos))):
                action = unit.transfer(direction_factory, 0, unit.cargo.ice, False)
                #print(f"{unit.unit_id} rb passing ice {unit.cargo.ice}")
            if unit.cargo.ore > unit.unit_cfg.DIG_RESOURCE_GAIN * 5 and\
                not all(unit_next_action(unit) == unit.move(direction_to(factory.pos, unit.pos))):
                action = unit.transfer(direction_factory, 1, unit.cargo.ore, False)
                #print(f"{unit.unit_id} rb passing ore {unit.cargo.ore}")
        elif not unit_on_factory(g_state, unit)[0]:
            if len(unit_log[unit.unit_id][0]) > 1:
                return_action, cost = log_calc(g_state, unit, unit_log[unit.unit_id][0])
                if unit.power < cost + unit.unit_cfg.MOVE_COST * 10 + unit.unit_cfg.DIG_COST and unit.power >= unit.unit_cfg.MOVE_COST:
                    action = return_action
                    #print(f"{unit.unit_id} remote return len {len(unit_log[unit.unit_id])-1} where {unit_log[unit.unit_id]}")
                elif unit.power >= unit.unit_cfg.DIG_COST and (resoure_exist(g_state, unit.pos, 0) or resoure_exist(g_state, unit.pos, 1)):
                    #すごく強くなるようならここを取る。
                    action = unit.dig(True)
                    #print(f"{unit.unit_id} remote dig pow {unit.power}")

        return action

    def water_adj_pos(g_state:GameState):
        adj_vecs = np.array([[2,0],[2,1],[2,-1],[-2,0],[-2,1],[-2,-1],[0,2],[1,2],[-1,2],[0,-2],[1,-2],[-1,-2]])
        ice_grids = []
        env_cfg = g_state.env_cfg
        for i in range(0,env_cfg.map_size):
            for j in range(0,env_cfg.map_size):
                if g_state.board.ice[j][i] == 1 :
                    ice_grids.append([i,j])
        return_grids = []
        for grid in ice_grids:
            for vec in adj_vecs:
                target_grid = [grid[0]+vec[0], grid[1]+vec[1]]
                if not(target_grid in return_grids) and not(pos_overrap_factory(g_state, np.array([i,j]))):
                    return_grids.append(target_grid)
        return np.array(return_grids)

    tokens = tokens.squeeze(0)

    actions = {}
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
                if factory.cargo.metal >= factory.build_light_metal_cost(state) and factory.power >= factory.build_light_power_cost(state)\
                    and state.env_steps > env_cfg.max_episode_length/3:
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
                if  action_value < 1:#直進
                    direction = unit_log[unit.unit_id][1]
                    cost = unit.move_cost(state, direction)
                    if not(cost == None) and unit.power >= cost:
                        action = unit.move(direction, True)
                elif action_value < 2:#右折
                    direction = unit_log[unit.unit_id][1] % 4 + 1
                    cost = unit.move_cost(state, direction)
                    if not(cost == None) and unit.power >= cost:
                        action = unit.move(direction, True)
                elif action_value < 3:#左折
                    direction = (unit_log[unit.unit_id][1] + 2) % 4 + 1
                    cost = unit.move_cost(state, direction)
                    if not(cost == None) and unit.power >= cost:
                        action = unit.move(direction, True)
                elif action_value < 4:#transferはrule_based
                    direction_dict = {1:[0,-1], 2:[1, 0], 3:[0,-1], 4:[-1,0]}
                    if unit.cargo.ice > 0 or unit.cargo.ore > 0:
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
                    else:
                        direction = unit_log[unit.unit_id][1]
                        cost = unit.move_cost(state, direction)
                        if not(cost == None) and unit.power >= cost:
                            action = unit.move(direction, True)
                elif action_value < 5:#pick_upはiceかoreのみ
                    if unit_on_factory(state, unit)[0]:
                        resource_type = int(((action_value-2) * 2) % 2)
                        action = unit.pickup(resource_type, 100, False)
                    elif unit.power >= unit.dig_cost(state):
                        action = unit.dig(False)
                elif action_value < 6:#digだが、自身破壊入れるならここ
                    is_on, factory_ = unit_on_factory(state, unit)
                    if is_on:
                        action = unit.pickup(4, min(unit.dig_cost(state)*5, factory_.power), True)
                    elif unit.power >= unit.dig_cost(state):
                        direction = unit_log[unit.unit_id][1]
                        cost = unit.move_cost(state, direction)
                        if rubble_num(state, unit.pos):
                            action = unit.dig(True)
                        elif not(cost == None) and unit.power >= unit.move_cost(state, direction):
                            action = unit.move(direction, True)
                else:
                    print("error-tipo")
            
            #print(f"`{unit.unit_id}, {action}, {unit_next_action(unit)}")
            if not (action is None) and not is_the_same_action(action, unit_next_action(unit)):
                #print("change")
                actions[unit.unit_id] = [action]

    elif state.env_steps != 0:
        def check_is_in(target_position:np.ndarray, position_array:np.ndarray):
            is_in = False
            for i in range(position_array.shape[0]):
                if all(target_position == position_array[i]) or \
                    (pos_on_factory(state, pos)[0] and f"player_{pos_on_factory(state, pos)[1].team_id}" == agent):
                    is_in = True
                    break
            return is_in
        pos = np.zeros(2)
        for i in range(2):
            action_value = 0
            embedder = start_embedder(i)
            for k in range(token_len):
                action_value += embedder[k]*tokens[k]
            grid = math.ceil(action_value * 48 % 48)
            pos[i] = grid
        potential_spawns:np.ndarray = state.board.spawns[agent]
        if not check_is_in(pos, potential_spawns):
            if check_is_in(np.array([pos[0], env_cfg.map_size - pos[1]]), potential_spawns):
                pos = np.array([pos[0], env_cfg.map_size - pos[1]])
            elif check_is_in(np.array([env_cfg.map_size - pos[0], pos[1]]), potential_spawns):
                pos = np.array([env_cfg.map_size - pos[0], pos[1]])
            else:
                print("no good pos")
        water_adjs:np.ndarray = water_adj_pos(state)
        water_potentials = []
        for i in range(water_adjs.shape[0]):
            for j in range(potential_spawns.shape[0]):
                if all(water_adjs[i] == potential_spawns[j]):
                    water_potentials.append(water_adjs[i])
        length = 100
        index = 0
        if len(water_adjs) > 0:
            for i in range(len(water_potentials)):
                if pos_overrap_factory(state, water_potentials[i]):
                    continue
                length_ = abs(water_potentials[i][0]-pos[0])+abs(water_potentials[i][1]-pos[1])
                if length_ < length:
                    index = i
                    length = length_
                actions = dict(spawn = water_potentials[index], metal = 100, water = 100)
        else:
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

    print(f"actions = {actions}")
    return actions

def env_to_tokens(state:GameState, unit_log, view_agent):#雑に作る。若干の情報のオーバーラップは仕方なし。
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
            direction_value = unit_log[unit.unit_id][1]/4
            unit_infos[x_index][y_index+8] = direction_value
            y_index += unit_info_dim
            if y_index+unit_info_dim > token_len:
                y_index = 0
                x_index += 1
    tokens = np.concatenate((tokens, unit_infos))
    #ラスト基本情報
    basics = np.zeros((1, token_len))
    basics[0][0] = state.real_env_steps / state.env_cfg.max_episode_length
    if state.real_env_steps < 0:
        pass
    elif state.env_cfg.max_episode_length - state.real_env_steps > token_len - 1:
        #print(state.weather_schedule[state.real_env_steps:state.real_env_steps+token_len-1])
        basics[0][1:] = state.weather_schedule[state.real_env_steps:state.real_env_steps+token_len-1]
    elif state.real_env_steps < 1000:
        #print(state.weather_schedule[state.real_env_steps:])
        basics[0][1:1+state.env_cfg.max_episode_length - state.real_env_steps] = state.weather_schedule[state.real_env_steps:]
    tokens = np.concatenate((tokens, basics))
    
    return tokens

def unit_on_factory(g_state:GameState, unit:Unit):
    team_name = f"player_{unit.team_id}"
    factories = g_state.factories[team_name].values()
    for factory in factories:
        unit_pos = unit.pos
        factory_pos = factory.pos
        if abs(unit_pos[0]-factory_pos[0]) < 2 and abs(unit_pos[1]-factory_pos[1]) < 2:
            return True, factory
    return False, None

#近傍アクション生成機
def action_nearby_token(token:np.ndarray, variance):
    random_array = 3 * np.random.rand(*token.shape)
    token_ = (1 - variance) * token + variance * random_array
    #print(f"token = {token}")
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
        self.v_net = ValueNet()
        self.a_net.to(device)
        self.a_net.load_state_dict(torch.load(a_model_path, map_location= device))
        self.v_net.to(device)
        self.v_net.load_state_dict(torch.load(v_model_path, map_location= device))
        self.post_state_value = 0
        self.unit_log = {}

    def determin_action(self, step:int, obs, remainingOverageTime: int = 60):
        def log_addition(log:Tuple, pos:np.ndarray):
            def check_is_inlog():
                exist = False
                index = -1
                for i in range(len(log[0])):
                    if all(pos == log[0][i]):
                        exist = True
                        index = i
                        break
                return exist, index

            exist, index = check_is_inlog()
            if exist:
                direction = 0
                if index == 0 and len(log[0])>1:
                    direction = direction_to(log[0][1], pos)
                else:
                    direction = direction_to(log[0][0], pos)
                return (log[0][index:], direction)
            else:
                log[0].insert(0, pos)
                return (log[0], direction_to(log[0][1], log[0][0]))
        state = obs_to_game_state(step, self.env_cfg, obs)
        my_units = state.units[self.player]
        for unit in my_units.values():
            is_on, factory = unit_on_factory(state, unit)
            if not(unit.unit_id in self.unit_log.keys()):
                self.unit_log[unit.unit_id] = ([unit.pos], random.randint(1, 4))
            elif is_on:
                direction = direction_to(factory.pos, unit.pos)
                if direction == 0:
                    direction = random.randint(1, 4)
                self.unit_log[unit.unit_id] = ([unit.pos], direction)
            else:
                self.unit_log[unit.unit_id] = log_addition(self.unit_log[unit.unit_id], unit.pos)
        state_tokens = torch.from_numpy(env_to_tokens(state, self.unit_log, self.player)).float().to(device)
        a_t = self.a_net(state_tokens)
        a = a_t.to('cpu').detach().numpy()
        value = self.v_net(torch.concat([state_tokens, a_t])).item()
        for i in range(10):#もちろん時間で区切ってもよし
            a_ = action_nearby_token(a, 0.1)
            a_t_ = torch.from_numpy(a_).float().to(device)
            value_ = self.v_net(torch.concat([state_tokens, a_t_])).item()
            if (value_ - value)/(abs(value)+1e-10) > 0:
                a = a_
                value = value_
        real_action = tokens_to_actions(state, a, self.player, self.unit_log)
        return real_action

    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
        return self.determin_action(step, obs, remainingOverageTime)

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        state = obs_to_game_state(step, self.env_cfg, obs)
        print("statevalue step {0} is {1:3g}, {2:3g}".format(state.real_env_steps, state_value(state, "player_0"), state_value(state, "player_1")))
        return self.determin_action(step, obs, remainingOverageTime)