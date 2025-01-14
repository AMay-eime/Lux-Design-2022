from lux.kit import obs_to_game_state, GameState, EnvConfig, Team, Factory, Unit, UnitCargo
from lux.utils import direction_to, is_the_same_action, secondery_directiopn_to, move_effective_direction_to, secondary_move_effective_direction_to
import numpy as np
import sys
from luxai_runner.utils import to_json
import json
from luxai2022 import LuxAI2022
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict
from torch.utils.data import DataLoader
import math
import random
import time

orth_adj_list = [np.array([0,1]), np.array([0,-1]), np.array([-1,0]), np.array([1,0])]
factory_area_list = [np.array([0,0]), np.array([0,1]), np.array([0,-1]), np.array([-1,0]), np.array([1,0]),\
    np.array([1,1]), np.array([1,-1]), np.array([-1,1]), np.array([-1,-1])]

#config
restart_epoch = 211

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
token_len = 288
env_token_num = 10
action_token_num = 2

heads = 0
encoder_layers = 6
decoder_layers = 6
epochs = 2000
swing_range = 100
results_per_epoch = 5000
batch_size = 16

beta_max = 0.3
beta_min = 0.1
gamma_min = 0.5
gamma_max = 0.98
advantage_steps = 10

#rule basedを制御する変数
target_light_num = 3
factory_territory = 2
least_water_storage = 500
tactical_check_step = 800

#盤面の評価に使える便利所たち
def distance(pos1:np.ndarray, pos2:np.ndarray):
    if type(pos1) == list:
        pos1 = np.array(pos1)
    if type(pos2) == list:
        pos2 = np.array(pos2)
    ds = pos2-pos1
    return abs(ds[0]) + abs(ds[1])

def pos_out_map(pos:np.ndarray, mapsize) -> bool:
    return pos[0] < 0 or mapsize-1 < pos[0] or pos[1] < 0 or mapsize-1 < pos[1]

def resoure_exist(g_state:GameState, pos:np.ndarray, resource_type):#type = 1(ice) 0(ore)
        ice_existance = g_state.board.ice[pos[0]][pos[1]]
        ore_existance = g_state.board.ore[pos[0]][pos[1]]
        if resource_type == 0:
            return ice_existance == 1
        elif resource_type == 1:
            return ore_existance == 1
        else:
            print("resouce what")

def rubble_num(g_state:GameState, pos:np.ndarray):
    return g_state.board.rubble[pos[0]][pos[1]]

def lichen_num(g_state:GameState, pos:np.ndarray):
    return g_state.board.lichen[pos[0]][pos[1]]

def strain_num(g_state:GameState, pos:np.ndarray):
    return g_state.board.lichen_strains[pos[0]][pos[1]]

def unit_id_num(unit:Unit):
    str_id = unit.unit_id
    return int(str_id.split("_")[1])

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

def get_lichen_dict(g_state:GameState, view_agent):
    opp_agent = "player_1" if view_agent == "player_0" else "player_0"
    my_factories = g_state.factories[view_agent]
    opp_factories = g_state.factories[opp_agent]
    my_strains = []
    opp_strains = []
    my_lichen_dict = {}
    opp_lichen_dict = {}
    for factory in my_factories.values():
        my_strains.append(factory.strain_id)
    for factory in opp_factories.values():
        opp_strains.append(factory.strain_id)
    lichen_map = g_state.board.lichen
    strain_map = g_state.board.lichen_strains
    for i in range(strain_map.shape[0]):
        for k in range(strain_map.shape[1]):
            strain_id = strain_map[i][k]
            if strain_id in my_strains:
                my_lichen_dict[np.array([i,k]).astype(np.int32).tobytes()] = lichen_map[i][k]
                if lichen_map[i][k] == 0:
                    print("error, lichen 0 added dict")
            if strain_id in opp_strains:
                opp_lichen_dict[np.array([i,k]).astype(np.int32).tobytes()] = lichen_map[i][k]
                if lichen_map[i][k] == 0:
                    print("error, lichen 0 added dict")
    return my_lichen_dict, opp_lichen_dict

def get_pseudo_lichen_dict(g_state:GameState, view_agent):
    opp_agent = "player_1" if view_agent == "player_0" else "player_0"
    my_factories = g_state.factories[view_agent]
    opp_factories = g_state.factories[opp_agent]
    my_strains = []
    opp_strains = []
    my_lichen_dict = {}
    opp_lichen_dict = {}
    factory_lichenpos_dict = {}
    pos_to_factory_dict = {}
    my_area_lichen = []
    my_area_none = []
    opp_area_lichen = []
    opp_area_none = []
    for factory in my_factories.values():
        my_strains.append(factory.strain_id)
        factory_lichenpos_dict[factory.unit_id] = []
        for grid in factory_area_list:
            if lichen_num(g_state, factory.pos + grid) == 0:
                my_area_none.append((factory.pos + grid).astype(np.int32).tobytes())
                pos_to_factory_dict[(factory.pos + grid).astype(np.int32).tobytes()] = factory.unit_id
            elif strain_num(g_state, factory.pos + grid) == factory.strain_id:
                my_area_lichen.append((factory.pos + grid).astype(np.int32).tobytes())
                pos_to_factory_dict[(factory.pos + grid).astype(np.int32).tobytes()] = factory.unit_id
    for factory in opp_factories.values():
        opp_strains.append(factory.strain_id)
        factory_lichenpos_dict[factory.unit_id] = []
        for grid in factory_area_list:
            if lichen_num(g_state, factory.pos + grid) == 0:
                opp_area_none.append((factory.pos + grid).astype(np.int32).tobytes())
                pos_to_factory_dict[(factory.pos + grid).astype(np.int32).tobytes()] = factory.unit_id
            elif strain_num(g_state, factory.pos + grid) == factory.strain_id:
                opp_area_lichen.append((factory.pos + grid).astype(np.int32).tobytes())
                pos_to_factory_dict[(factory.pos + grid).astype(np.int32).tobytes()] = factory.unit_id
    my_temp_area = []
    opp_temp_area = []
    pseudo_lichen = 0
    while len(my_area_lichen) > 0 or len(opp_area_lichen) > 0:
        for grid_byte in my_area_lichen:
            my_lichen_dict[grid_byte] = (True, pseudo_lichen)
        for grid_byte in opp_area_lichen:
            opp_lichen_dict[grid_byte] = (True, pseudo_lichen)
        for grid_byte in my_area_lichen:
            for vec in orth_adj_list:
                grid = np.frombuffer(grid_byte, dtype = np.int32)
                target_grid:np.ndarray = grid + vec
                if pos_out_map(target_grid, g_state.env_cfg.map_size) or rubble_num(g_state, target_grid) > 0 or resoure_exist(g_state, target_grid, 0) or resoure_exist(g_state, target_grid, 1):
                    continue
                if not(target_grid.astype(np.int32).tobytes() in my_lichen_dict) and not(target_grid.astype(np.int32).tobytes() in opp_lichen_dict) and\
                    not(target_grid.astype(np.int32).tobytes() in my_temp_area) and not(target_grid.astype(np.int32).tobytes() in opp_temp_area):
                    if lichen_num(g_state, target_grid) > 0 and strain_num(g_state, target_grid) == strain_num(g_state, grid):
                        my_temp_area.append(target_grid.astype(np.int32).tobytes())
                        factory_lichenpos_dict[pos_to_factory_dict[grid_byte]].append(target_grid)
                        pos_to_factory_dict[target_grid.astype(np.int32).tobytes()] = pos_to_factory_dict[grid_byte]
                    elif lichen_num(g_state, target_grid) == 0:
                        my_area_none.append(target_grid.astype(np.int32).tobytes())
                        factory_lichenpos_dict[pos_to_factory_dict[grid_byte]].append(target_grid)
                        pos_to_factory_dict[target_grid.astype(np.int32).tobytes()] = pos_to_factory_dict[grid_byte]
        for grid_byte in opp_area_lichen:
            for vec in orth_adj_list:
                target_grid:np.ndarray = np.frombuffer(grid_byte, dtype = np.int32) + vec
                if pos_out_map(target_grid, g_state.env_cfg.map_size) or rubble_num(g_state, target_grid) > 0 or resoure_exist(g_state, target_grid, 0) or resoure_exist(g_state, target_grid, 1):
                    continue
                if not(target_grid.astype(np.int32).tobytes() in my_lichen_dict) and not(target_grid.astype(np.int32).tobytes() in opp_lichen_dict) and\
                    not(target_grid.astype(np.int32).tobytes() in my_temp_area) and not(target_grid.astype(np.int32).tobytes() in opp_temp_area):
                    if lichen_num(g_state, target_grid) > 0 and strain_num(g_state, target_grid) == strain_num(g_state, grid):
                        opp_temp_area.append(target_grid.astype(np.int32).tobytes())
                        factory_lichenpos_dict[pos_to_factory_dict[grid_byte]].append(target_grid)
                        pos_to_factory_dict[target_grid.astype(np.int32).tobytes()] = pos_to_factory_dict[grid_byte]
                    elif lichen_num(g_state, target_grid) == 0:
                        opp_area_none.append(target_grid.astype(np.int32).tobytes())
                        factory_lichenpos_dict[pos_to_factory_dict[grid_byte]].append(target_grid)
                        pos_to_factory_dict[target_grid.astype(np.int32).tobytes()] = pos_to_factory_dict[grid_byte]
        my_area_lichen = my_temp_area.copy()
        opp_area_lichen = opp_temp_area.copy()
        my_temp_area = []
        opp_temp_area = []
        pseudo_lichen += 1
    pseudo_lichen = 0
    while len(my_area_none) > 0 or len(opp_area_none) > 0:
        for grid_byte in my_area_none:
            my_lichen_dict[grid_byte] = (False, pseudo_lichen)
        for grid_byte in opp_area_none:
            opp_lichen_dict[grid_byte] = (False, pseudo_lichen)
        for grid_byte in my_area_none:
            for vec in orth_adj_list:
                target_grid:np.ndarray = np.frombuffer(grid_byte, dtype = np.int32) + vec
                if pos_out_map(target_grid, g_state.env_cfg.map_size) or rubble_num(g_state, target_grid) > 0 or resoure_exist(g_state, target_grid, 0) or resoure_exist(g_state, target_grid, 1):
                    continue
                if not(target_grid.astype(np.int32).tobytes() in my_lichen_dict) and not(target_grid.astype(np.int32).tobytes() in opp_lichen_dict) and\
                    not(target_grid.astype(np.int32).tobytes() in my_temp_area) and not(target_grid.astype(np.int32).tobytes() in opp_temp_area):
                    my_temp_area.append(target_grid.astype(np.int32).tobytes())
                    factory_lichenpos_dict[pos_to_factory_dict[grid_byte]].append(target_grid)
                    pos_to_factory_dict[target_grid.astype(np.int32).tobytes()] = pos_to_factory_dict[grid_byte]
        for grid_byte in opp_area_none:
            for vec in orth_adj_list:
                target_grid:np.ndarray = np.frombuffer(grid_byte, dtype = np.int32) + vec
                if pos_out_map(target_grid, g_state.env_cfg.map_size) or rubble_num(g_state, target_grid) > 0 or resoure_exist(g_state, target_grid, 0) or resoure_exist(g_state, target_grid, 1):
                    continue
                if not(target_grid.astype(np.int32).tobytes() in my_lichen_dict) and not(target_grid.astype(np.int32).tobytes() in opp_lichen_dict) and\
                    not(target_grid.astype(np.int32).tobytes() in my_temp_area) and not(target_grid.astype(np.int32).tobytes() in opp_temp_area):
                    opp_temp_area.append(target_grid.astype(np.int32).tobytes())
                    factory_lichenpos_dict[pos_to_factory_dict[grid_byte]].append(target_grid)
                    pos_to_factory_dict[target_grid.astype(np.int32).tobytes()] = pos_to_factory_dict[grid_byte]
        my_area_none = my_temp_area.copy()
        opp_area_none = opp_temp_area.copy()
        my_temp_area = []
        opp_temp_area = []
        pseudo_lichen += 1

    return my_lichen_dict, opp_lichen_dict, factory_lichenpos_dict

def get_tactical_points(g_state:GameState, my_lichen, opp_lichen, my_factories):
    def more_distal(src, tgt):
        return (tgt[0] and not src[0]) or (src[1] > tgt[1])
    dig_pos = []
    for pos_byte, lichen_tp in my_lichen.items():
        pos:np.ndarray = np.frombuffer(pos_byte, dtype = np.int32)
        lichen_exist, lichen = lichen_tp
        if lichen < 50:
            #自分の植物の周りで、上下左右を走査。rubble一個挟んで空地ならrubble地をtacticalpointとする。
            #その際、自分の周囲に敵の土地があったらダメです。
            opp_lichen_is = False
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if i == 0 and j == 0:
                        continue
                    target_pos = pos + np.array([i,j])
                    if target_pos.tobytes() in opp_lichen:
                        opp_lichen_is = True
            if opp_lichen_is:
                continue
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if (i == 0 or j == 0) and not(i == 0 and j == 0):
                        target_pos = pos + np.array([i,j])
                        if pos_out_map(target_pos, g_state.env_cfg.map_size):
                            continue
                        if 0 < rubble_num(g_state, target_pos) and rubble_num(g_state, target_pos) < 30:
                            check_pos:np.ndarray = pos + 2*np.array([i,j])
                            if pos_out_map(check_pos, g_state.env_cfg.map_size):
                                continue
                            if rubble_num(g_state, check_pos) == 0 and (not(check_pos.astype(np.int32).tobytes() in my_lichen) or lichen < my_lichen[check_pos.astype(np.int32).tobytes()]-10)and\
                                (not(check_pos.astype(np.int32).tobytes() in opp_lichen) or lichen < opp_lichen[check_pos.astype(np.int32).tobytes()]-10):
                                #壁の向こう側がインタクト、または支配地でも遠い場合
                                dig_pos.append(target_pos)
    secondary_pos = [] 
    secondary_lichen_dict = {}
    if g_state.real_env_steps < tactical_check_step:
        resource_pos_list = []
        for i in range(g_state.env_cfg.map_size):
            for j in range(g_state.env_cfg.map_size):
                grid_ = np.array([i,j])
                if resoure_exist(g_state, grid_, 0) or resoure_exist(g_state, grid_, 1):
                    resource_pos_list.append(grid_)
        temp_list = []
        for _, factory_ in my_factories.items():
            pos_dist_dict = {}
            for pos_ in resource_pos_list:
                max_item = (0, 100)
                if len(pos_dist_dict) > 0:
                    max_item = max(pos_dist_dict.items(), key = lambda x:x[1])
                dist = distance(factory_.pos, pos_)
                near_factory = False
                for _, factory_a in my_factories.items():
                    if distance(factory_a.pos, pos_) < factory_territory * 3:
                        near_factory = True
                        break
                if dist < max_item[1] and not near_factory:
                    pos_dist_dict[pos_.astype(np.int32).tobytes()] = dist
                if len(pos_dist_dict) > 3:
                    pos_dist_dict.pop(max_item[0])
            for pos_byte in pos_dist_dict.keys():
                if not pos_byte in temp_list:
                    temp_list.append(pos_byte)
        for pos_byte in temp_list:
            secondary_pos.append(np.frombuffer(pos_byte, dtype=np.int32))
    else:
        for pos_byte, lichen_tp in opp_lichen.items():
            pos = np.frombuffer(pos_byte, dtype = np.int32)
            lichen_exist, lichen = lichen_tp
            if lichen_exist or lichen < 25:#あまり遠すぎても効果薄いし計算時間かかる
                zero_lichen_check_list = [[2,2],[-2,2],[2,-2],[-2,-2]]
                same_lichen_is = False
                one_more_pseudo_lichens = []
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if i == 0 and j == 0:
                            continue
                        target_pos = pos + np.array([i,j])
                        target_pos_byte = target_pos.astype(np.int32).tobytes()
                        if target_pos_byte in opp_lichen and opp_lichen[target_pos_byte][0] == lichen_exist and opp_lichen[target_pos_byte][1] == lichen:
                            same_lichen_is = True
                            continue
                        if target_pos_byte in opp_lichen and more_distal(opp_lichen[target_pos_byte], lichen_tp):
                            one_more_pseudo_lichens.append(target_pos)
                if same_lichen_is or len(one_more_pseudo_lichens) == 0:
                    continue
                past_pos = one_more_pseudo_lichens.copy()
                past_pos.append(pos)
                now_pos = one_more_pseudo_lichens.copy()
                temp_pos = []
                link_num = 0
                while link_num < 10 and len(now_pos) > 0:
                    for pos_ in now_pos:
                        pos_byte_ = pos_.astype(np.int32).tobytes()
                        link_num += 1
                        for dir in orth_adj_list:
                            target_pos = pos_ + dir
                            target_pos_byte = target_pos.astype(np.int32).tobytes()
                            if pos_out_map(target_pos, g_state.env_cfg.map_size):
                                continue
                            already_is = False
                            for pas_pos_ in past_pos:
                                if all(pas_pos_ == target_pos):
                                    already_is = True
                                    break
                            if not already_is and target_pos_byte in opp_lichen and more_distal(opp_lichen[target_pos_byte], opp_lichen[pos_byte_]):
                                temp_pos.append(target_pos)
                                past_pos.append(target_pos)
                    now_pos = temp_pos.copy()
                    temp_pos = []
                if not link_num < 10:
                    secondary_lichen_dict[pos.astype(np.int32).tobytes()] = lichen_tp
        for pos_byte, lichen_tp in secondary_lichen_dict.items():
            better_exists = False
            for pos_byte_ in secondary_lichen_dict:
                pos = np.frombuffer(pos_byte, dtype=np.int32) 
                pos_ = np.frombuffer(pos_byte_, dtype=np.int32)
                if distance(pos, pos_) == 1 and more_distal(secondary_lichen_dict[pos_byte], secondary_lichen_dict[pos_byte_]):
                    better_exists = True
                    break
            if not better_exists:
                secondary_pos.append(np.frombuffer(pos_byte, dtype=np.int32))
    return dig_pos, secondary_pos

def get_chase_points(g_state:GameState, view_agent):
    opp_agent = "player_1" if view_agent == "player_0" else "player_0"
    opp_units = g_state.units[opp_agent]
    return_list = []
    for _, opp_unit in opp_units.items():
        if resoure_exist(g_state, opp_unit.pos, 0) or  resoure_exist(g_state, opp_unit.pos, 1):
            return_list.append(opp_unit.pos)
    return return_list

def path_finding_direction_to(g_state:GameState, src:np.ndarray, tgt:np.ndarray, team_id:int, initial_direction:int = 0, avoid_opp_heavy = True):
    if type(src) == list:
        src = np.array(src)
    if type(tgt) == list:
        tgt = np.array(tgt)
    if distance(src, tgt) == 0:
        #print("path finding same grids")
        return 0, 0
    board:np.ndarray = np.zeros((g_state.env_cfg.map_size, g_state.env_cfg.map_size))
    opp_factories = g_state.factories[f"player_{1-team_id}"]
    my_units = g_state.units[f"player_{team_id}"]
    opp_units = g_state.units[f"player_{1-team_id}"]
    for _, factory in opp_factories.items():
        f_pos = factory.pos
        for vec in factory_area_list:
            grid = f_pos + np.array(vec)
            board[grid[0]][grid[1]] = 1
    for _, unit in my_units.items():
        if distance(src, unit.pos) > 0:
            board[unit.pos[0]][unit.pos[1]] = 1
            if distance(src, unit.pos) == 2:
                for vec in orth_adj_list:
                    grid = unit.pos + np.array(vec)
                    if not pos_out_map(grid, g_state.env_cfg.map_size) and not distance(src, grid) == 0:
                        board[grid[0]][grid[1]] = 1
    for _, unit in opp_units.items():
        if distance(src, unit.pos) > 0 and unit.unit_type == "HEAVY" and avoid_opp_heavy:
            for vec in orth_adj_list:
                if not pos_out_map(unit.pos + vec, g_state.env_cfg.map_size) and not distance(src, unit.pos + vec) == 0:
                    board[unit.pos[0] + vec[0]][unit.pos[1] + vec[1]] = 1
    dist = 0
    distance_pos_dict = {dist:[(tgt, 0)]}
    already_searched_list = [tgt.astype(np.int32).tobytes()]
    direction_dict = {0:[0,0], 1:[0,-1], 2:[1, 0], 3:[0,1], 4:[-1,0]}
    direction = 0
    def check_reached_src(pos:np.ndarray, last_direction):
        if distance(pos, src) == 0:
            return True
        if not initial_direction == 0 and abs(initial_direction - last_direction) == 2 and distance(pos+np.array(direction_dict[last_direction]), src) == 0:
            return True
        return False
    while dist in distance_pos_dict and len(distance_pos_dict[dist]) > 0 and direction == 0:
        for grid, dir_ in distance_pos_dict[dist]:
            if dir_ == 0:
                for move_num, vec in direction_dict.items():
                    check_grid:np.ndarray = grid + np.array(vec)
                    if pos_out_map(check_grid, g_state.env_cfg.map_size) or check_grid.astype(np.int32).tobytes() in already_searched_list\
                        or board[check_grid[0]][check_grid[1]] == 1:
                        continue
                    if check_reached_src(check_grid, move_num):
                        direction = (move_num + 1)%4 + 1
                        break
                    already_searched_list.append(check_grid.astype(np.int32).tobytes())
                    if dist+1 in distance_pos_dict:
                        distance_pos_dict[dist+1].append((check_grid, move_num))
                    else:
                        distance_pos_dict[dist+1] = [(check_grid, move_num)]
            else:
                move_num = dir_
                vec = direction_dict[move_num]
                check_grid = grid + np.array(vec)
                if not pos_out_map(check_grid, g_state.env_cfg.map_size) and not check_grid.astype(np.int32).tobytes() in already_searched_list\
                    and board[check_grid[0]][check_grid[1]] == 0:
                    if check_reached_src(check_grid, move_num):
                        direction = (move_num + 1)%4 + 1
                        break
                    already_searched_list.append(check_grid.astype(np.int32).tobytes())
                    if dist+1 in distance_pos_dict:
                        distance_pos_dict[dist+1].append((check_grid, move_num))
                    else:
                        distance_pos_dict[dist+1] = [(check_grid, move_num)]
        if dist > 0:
            for grid, dir_ in distance_pos_dict[dist-1]:
                for move_num, vec in direction_dict.items():
                    check_grid = grid + np.array(vec)
                    if pos_out_map(check_grid, g_state.env_cfg.map_size) or check_grid.astype(np.int32).tobytes() in already_searched_list\
                    or board[check_grid[0]][check_grid[1]] == 1:
                        continue
                    if check_reached_src(check_grid, move_num):
                        direction = (move_num + 1)%4 + 1
                        break
                    already_searched_list.append(check_grid.astype(np.int32).tobytes())
                    if dist+1 in distance_pos_dict:
                        distance_pos_dict[dist+1].append((check_grid, move_num))
                    else:
                        distance_pos_dict[dist+1] = [(check_grid, move_num)]
        dist += 1
    return direction, dist

def factory_output_linkage(g_state:GameState, factory_pos:np.ndarray, direction:int):
    direction_dict = {0:[0,0], 1:[0,-1], 2:[1, 0], 3:[0,1], 4:[-1,0]}
    out_pos = factory_pos + np.array(direction_dict[direction]) * 2
    linkage_pos = []
    for i in range(1,5):
        if not i == direction:
            pos_ = factory_pos + np.array(direction_dict[i]) * 2
            linkage_pos.append(pos_.astype(np.int32).tobytes())
    post_pos = []
    now_pos = []
    temp_pos = []
    linked = False
    search_distance = 0
    for vec in factory_area_list:
        post_pos.append((factory_pos + vec).astype(np.int32).tobytes())
    post_pos.append(out_pos.astype(np.int32).tobytes())
    now_pos.append(out_pos)
    while not linked and len(now_pos) > 0 and search_distance < 30:
        search_distance += 1
        for pos_ in now_pos:
            for vec in orth_adj_list:
                target_pos:np.ndarray = pos_ + vec
                if pos_out_map(target_pos, g_state.env_cfg.map_size) or target_pos.astype(np.int32).tobytes() in post_pos:
                    continue
                if target_pos.astype(np.int32).tobytes() in linkage_pos:
                    linked = True
                    break
                if rubble_num(g_state, target_pos) == 0:
                    temp_pos.append(target_pos)
                    post_pos.append(target_pos.astype(np.int32).tobytes())
        now_pos = temp_pos.copy()
        temp_pos = []
    return linked

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
            if distance(pos, factory_pos) < 5 + factory_territory * 2 and\
                abs(pos[0]-factory_pos[0]) < 3 + factory_territory*2 and abs(pos[1]-factory_pos[1]) < 3 + factory_territory*2:
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

    def unit_in_factory_territory(g_state:GameState, unit:Unit):
        team_name = f"player_{unit.team_id}"
        factories = g_state.factories[team_name].values()
        for factory in factories:
            unit_pos = unit.pos
            factory_pos = factory.pos
            if distance(unit_pos, factory_pos) < 3 + factory_territory and\
                abs(unit_pos[0]-factory_pos[0]) < 2 + factory_territory and abs(unit_pos[1]-factory_pos[1]) < 2 + factory_territory:
                return True, factory
        return False, None

    def unit_next_action(unit:Unit):
        if(len(unit.action_queue) == 0):
            return [0,0,0,0,0]
        else:
            return unit.action_queue[0]

    def rulebased_factory(g_state:GameState, factory:Factory, factory_pos_dict):
        action = None
        overrap = False
        for unit_ in my_units.values():
            if distance(unit_.pos, factory.pos) == 0:
                overrap = True
                break
        if factory.cargo.water - (env_cfg.max_episode_length - g_state.real_env_steps) >\
            max(factory.water_cost(g_state), len(factory_pos_dict[factory.unit_id]) / 40) * (env_cfg.max_episode_length - g_state.real_env_steps) and\
                len(factory_pos_dict[factory.unit_id]) > 0:
            #print("rb watering")
            action = factory.water()
        elif factory.cargo.metal >= factory.build_heavy_metal_cost(state) and factory.power >= factory.build_heavy_power_cost(state) and not overrap:
            action = factory.build_heavy()
        elif factory.cargo.metal >= factory.build_light_metal_cost(state) and factory.power >= factory.build_light_power_cost(state) and not overrap:
            #print(f"{g_state.real_env_steps} rb light")
            action = factory.build_light()
        return action

    def log_calc(g_state:GameState, unit:Unit, log:list):
        unit_cfg = unit.unit_cfg
        if len(log) < 2:
            print("error, not much log len")
            direction = 0
        else:
            direction = direction_to(log[0], log[1])
        return_cost = 0
        return_cost += (len(log)-1) * unit_cfg.MOVE_COST
        for i in range(1, len(log) - 1):
            pos = log[i]
            return_cost += rubble_num(g_state, pos) * unit_cfg.RUBBLE_MOVEMENT_COST
        
        return unit.move(direction, repeat=-1), return_cost

    def rulebased_unit(g_state:GameState, unit:Unit, tactical_points, factory_pos_dict, chase_pos):
        my_team_id = "player_0" if unit.team_id == 0 else "player_1"
        enemy_team_id = "player_1" if unit.team_id == 0 else "player_0"
        action = None
        adj, factory = unit_adjascent_factory(g_state, unit)
        is_on, factory_on = unit_on_factory(g_state, unit)
        exist, factory_base = pos_on_factory(g_state, unit_log[unit.unit_id][0][-1])
        in_terr, factory_terr = unit_in_factory_territory(g_state, unit)
        opp_units = g_state.units[enemy_team_id]

        if unit.unit_type == "LIGHT":#Light robotに割くエネルギーはほぼない。作戦行動用。
            search_list = np.array([[1,0],[-1,0],[0,1],[0,-1]])
            
            heavy_list = []
            my_light_list = []
            for unit_ in my_units.values():
                if unit_.unit_type == "HEAVY":
                    heavy_list.append(unit_.pos)
                if unit_.unit_type == "LIGHT":
                    my_light_list.append(unit_)
            for unit_ in opp_units.values():
                if unit_.unit_type == "HEAVY":
                    heavy_list.append(unit_.pos)

            #作戦行動、destruction優先で、残っているポイントのうち一番近いところへ行く
            #if len(tactical_points) > 0:
            #    print(f"{g_state.real_env_steps} tacticalpos = {tactical_points}")
            for d_pos in tactical_points:
                my_dist = distance(d_pos, unit.pos)
                nearest = True
                for l_rob in my_light_list:
                    l_pos:np.ndarray = l_rob.pos
                    at_another_pos = False
                    for d_pos_ in tactical_points:
                        if distance(d_pos_, l_pos) == 0 and distance(d_pos, d_pos_) > 0:
                            at_another_pos = True
                            break
                    if (distance(d_pos, l_pos) < my_dist or (distance(d_pos, l_pos) == my_dist and unit_id_num(unit) < unit_id_num(l_rob))) and\
                        distance(l_pos, unit.pos) > 0 and not at_another_pos:
                        nearest = False
                        break
                if nearest:
                    if my_dist > 0:
                        dir_path = path_finding_direction_to(g_state, unit.pos, d_pos, unit.team_id)[0]
                        if not dir_path == 0:
                            action = unit.move(path_finding_direction_to(g_state, unit.pos, d_pos, unit.team_id)[0], repeat=-1)
                        continue
                    else:
                        attack_direction = 0 
                        for _, unit_ in opp_units.items():
                            if distance(unit_.pos, unit.pos) == 1 and unit_.unit_type == "LIGHT":
                                attack_direction = direction_to(unit.pos, unit_.pos)
                                break
                        if attack_direction == 0:
                            if rubble_num(g_state, unit.pos) > 0 or resoure_exist(g_state, unit.pos, 0) or resoure_exist(g_state, unit.pos, 1):
                                action = unit.dig(repeat=-1)
                                break
                            elif lichen_num(g_state, unit.pos) > g_state.env_cfg.max_episode_length - g_state.real_env_steps:
                                action = unit.self_destruct()
                                break
                            else:
                                action = unit.move(0, repeat=-1)
                                #print(f"{g_state.real_env_steps} {unit.unit_id} is on tactical and wait")
                                break
                        else:
                            #print(f"{g_state.real_env_steps} {unit.unit_id} is disturbed")
                            action = unit.move(attack_direction, repeat=-1)
            if is_on and unit.power < unit.unit_cfg.BATTERY_CAPACITY-10:
                action = unit.pickup(4, min(factory_on.power, unit.unit_cfg.BATTERY_CAPACITY - unit.power))
            home_direction = 0 
            pow_cost = 0
            if exist: 
                home_direction, pow_cost = path_finding_direction_to(g_state, unit.pos, factory_base.pos, unit.team_id)
            if (action is None and (home_direction == 0 or pow_cost * 1.1 < unit.power)):
                #第二候補として、敵のlight_robotでかつ資源上にいるものがいればそれを対象に突進するが、追いかけっこ可能なエネルギーは残しておく
                for d_pos in chase_pos:
                    my_dist = distance(d_pos, unit.pos)
                    nearest = True
                    for l_rob in my_light_list:
                        l_pos:np.ndarray = l_rob.pos
                        at_tactical_pos = False
                        for t_pos in tactical_points:
                            if distance(t_pos, l_pos) == 0 and distance(d_pos, t_pos) > 0:
                                at_tactical_pos = True
                                break
                        if (distance(d_pos, l_pos) < my_dist or (distance(d_pos, l_pos) == my_dist and unit_id_num(unit) < unit_id_num(l_rob))) and\
                            distance(l_pos, unit.pos) > 0 and not at_tactical_pos:
                            nearest = False
                            break
                    if nearest:
                        action = unit.move(path_finding_direction_to(g_state, unit.pos, d_pos, unit.team_id)[0], repeat=-1)
            if (action is None):
                #攻撃対象がいないまたはエネルギーが少ない場合。これは生存本能になる。
                if adj and not unit.team_id == factory.team_id:
                    action = unit.move(direction_to(factory.pos, unit.pos), repeat=-1)
                for _, unit_ in opp_units.items():
                    if unit_.unit_type == "HEAVY" and distance(unit.pos, unit_.pos) == 1:
                        if not home_direction == 0:
                            action = unit.move(home_direction, repeat=-1)
                        else:
                            action = unit.move(direction_to(unit_.pos, unit.pos), repeat=-1)
                    if unit_.unit_type == "LIGHT" and distance(unit.pos, unit_.pos) == 1:
                        if unit_.power > unit.power and pow_cost * 1.1 < unit.power and not home_direction == 0:
                            action = unit.move(home_direction, repeat=-1)
                            break
                        elif not pos_on_factory(g_state, unit_.pos):
                            #print(f"{g_state.real_env_steps} {unit.unit_id} hanted too far")
                            action = unit.move(direction_to(unit.pos, unit_.pos), repeat=-1)
                            break
                        else:
                            #print(f"{g_state.real_env_steps} {unit.unit_id} hanting")
                            action = unit.move(direction_to(unit_.pos, unit.pos), repeat=-1)
            if action is None and exist and unit.power >= pow_cost * 2 and unit.power > unit.dig_cost(g_state):
                #何もやることがなさ過ぎて暇なら領土拡大を図る
                possible_grids = []
                nearest_heavy = None
                length = 10
                for _, unit_ in my_units.items():
                    if unit_.unit_type == "HEAVY" and distance(unit_.pos, factory_base.pos) < length:
                        nearest_heavy = unit_
                        length = distance(unit_.pos, factory_base.pos)
                factory_grids = factory_pos_dict[factory_base.unit_id]
                if len(factory_grids) == 0:
                    for grid in factory_adj_grids(factory_base, g_state.env_cfg):
                        possible_grids.append(grid)
                else:
                    for grid in factory_grids:
                        grid_byte = grid.astype(np.int32).tobytes()
                        if grid_byte in my_lichen:
                            for vec in orth_adj_list:
                                target_grid = grid + vec
                                if not pos_out_map(target_grid, g_state.env_cfg.map_size) and rubble_num(g_state, target_grid) < 100 - my_lichen[grid_byte] * 10 and\
                                    (nearest_heavy is None or distance(nearest_heavy.pos, target_grid) > 3) and 0 < rubble_num(g_state, target_grid):
                                    possible_grids.append(target_grid)
                if len(possible_grids) > 0:
                    now_dist = 100
                    target_pos = np.array([-1,-1])
                    for f_pos in possible_grids:
                        my_dist = distance(f_pos, unit.pos)
                        nearest = True
                        for l_rob in my_light_list:
                            l_pos:np.ndarray = l_rob.pos
                            if (distance(f_pos, l_pos) < my_dist or (distance(f_pos, l_pos) == my_dist and unit_id_num(unit) < unit_id_num(l_rob))) and\
                                distance(l_pos, unit.pos) > 0:
                                nearest = False
                                break
                        if nearest and my_dist < now_dist:
                            now_dist = my_dist
                            target_pos = f_pos
                    if 0 < now_dist and now_dist < 30:
                        #print(f"{g_state.real_env_steps} {unit.unit_id} go to {target_pos} from {unit.pos}")
                        action = unit.move(path_finding_direction_to(g_state, unit.pos, target_pos, unit.team_id)[0], repeat=-1)
                    elif now_dist == 0:
                        #print(f"{g_state.real_env_steps} {unit.unit_id} dig at {target_pos} cost {pow_cost}, pow {unit.power}")
                        action = unit.dig(repeat=-1)
            if action is None and not home_direction == 0 and pow_cost < unit.power and unit.power < pow_cost * 2:
                #可能なら帰巣
                #print(f"{g_state.real_env_steps} {unit.unit_id} returning home direction {home_direction}, power {unit.power}, cost {pow_cost}")
                action = unit.move(home_direction, repeat=-1)
            if (action is None):
                action = unit.move(0, repeat=-1)
            #生存本能(隣接避け、多少のクラッシュは仕方なし(学習しろ))
            target_pos = None
            for i in range(search_list.shape[0]):
                pos = unit.pos + search_list[i]
                for pos_ in heavy_list:
                    if all(pos == pos_):
                        target_pos = pos
                        break
                if not(target_pos is None):
                    break
            if not(target_pos is None) and not is_on:
                if not home_direction == 0:
                    action = unit.move(home_direction, repeat=-1)
                else:
                    action = unit.move(direction_to(unit_.pos, unit.pos), repeat=-1)

        if unit.unit_type == "HEAVY":
            if rubble_num(state, unit.pos) > 0:#隣に敵がいない時足元にrubbleがあれば採掘します。
                action = unit.dig(repeat=-1)
            if in_terr and exist and not(factory_terr.unit_id == factory_base.unit_id):#所属外のfactory範囲内であれば引き返します。
                action = log_calc(state, unit, unit_log[unit.unit_id][0])[0]
        if unit.unit_type == "HEAVY" and exist and (factory_base.cargo.water < env_cfg.max_episode_length - state.real_env_steps + len(factory_pos_dict[factory_base.unit_id]) * 10):
            #初期生産でかつ水資源に余裕がなければ所属ファクトリー周辺の水資源を探す(生存本能の次)。場合によっては一生水を確保し続けることに。
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
                    if resoure_exist(g_state, target_pos, 0):
                        ice_pos = target_pos
                        break
            if not (ice_pos is None) and unit.power > unit.dig_cost(g_state):
                if (distance(ice_pos, unit.pos) < 3 and rubble_num(g_state, unit.pos) > 0) or distance(ice_pos, unit.pos) == 0:
                    action = unit.dig()
                else:
                    #print(f"{unit.unit_id} is assigned {factory_base.unit_id} found ice at {ice_pos}")
                    action = unit.move(direction_to(unit.pos, ice_pos), repeat=-1)
        elif unit.unit_type == "HEAVY" and exist:
            total_light_num = 0
            on_base = False
            for unit_ in my_units.values():
                if unit_.unit_type == "LIGHT":
                    total_light_num += 1
                if distance(unit_.pos, factory_base.pos) == 0:
                    on_base = True
            if (total_light_num < len(tactical_points) + len(chase_pos) or not on_base) and factory_base.cargo.metal < factory_base.build_heavy_metal_cost(g_state):#場に自分のlight_unitが一定数以下しか存在しない場合は自身の周りにoreがある場合に掘りに行く(優先度さらに低)
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
                if not (ore_pos is None) and rubble_num(g_state, unit.pos) == 0:
                    action = unit.move(direction_to(unit.pos, ore_pos), repeat=-1)

        if is_on and unit.power < unit.unit_cfg.DIG_COST * 3:
            #print(f"{unit.unit_id} rb pickup")
            action = unit.pickup(4, min(unit.unit_cfg.BATTERY_CAPACITY - unit.power, factory_on.power), repeat=-1)
        elif adj:
            direction_factory = direction_to(unit.pos, factory.pos)
            if unit.power < unit.dig_cost(g_state) * 2 + unit.move_cost(g_state, direction_factory):
                action = unit.move(direction_factory, repeat=-1)
                #print(f"{unit.unit_id} rb move dir = {direction_factory} from {unit.pos} to {factory.pos}")
            elif unit.power >= unit.dig_cost(g_state) + unit.move_cost(g_state, direction_factory):
                pos = unit.pos
                if (resoure_exist(g_state, pos, 0) or resoure_exist(g_state, pos, 1)) and unit.unit_type == "HEAVY":
                    #print(f"{unit.unit_id} rb dig rubble = {rubble_num(g_state, pos)} pos = {pos}, ice = {unit.cargo.ice}, ore = {unit.cargo.ore}")
                    #print(f"ice = {resoure_exist(g_state, pos, 0)}, ore = {resoure_exist(g_state, pos, 1)}")
                    action = unit.dig(repeat=-1)
            if unit.cargo.ice > unit.unit_cfg.DIG_RESOURCE_GAIN * 5 and\
                not is_the_same_action(unit_next_action(unit), unit.move(direction_to(factory.pos, unit.pos), repeat=-1)):
                action = unit.transfer(direction_factory, 0, unit.cargo.ice, repeat=0)
            if unit.cargo.ore > unit.unit_cfg.DIG_RESOURCE_GAIN * 5 and factory.cargo.water > (env_cfg.max_episode_length - g_state.real_env_steps) and\
                not is_the_same_action(unit_next_action(unit), unit.move(direction_to(factory.pos, unit.pos), repeat=-1)):
                action = unit.transfer(direction_factory, 1, unit.cargo.ore, repeat=0)
        elif not unit_on_factory(g_state, unit)[0] and (unit.unit_type == "HEAVY" or g_state.real_env_steps < tactical_check_step) and exist:
            if len(unit_log[unit.unit_id][0]) > 1:
                return_action, cost = log_calc(g_state, unit, unit_log[unit.unit_id][0])
                home_direction, pow_cost = path_finding_direction_to(g_state, unit.pos, factory_base.pos, unit.team_id)
                if unit.power < cost + unit.unit_cfg.MOVE_COST * 10 + unit.unit_cfg.DIG_COST and unit.power >= unit.unit_cfg.MOVE_COST:
                    if not home_direction == 0 and unit.unit_type == "LIGHT":
                        action = unit.move(home_direction, repeat = -1)
                    elif unit.unit_type == "HEAVY":
                        action = return_action
                elif unit.power >= unit.unit_cfg.DIG_COST and (resoure_exist(g_state, unit.pos, 0) or resoure_exist(g_state, unit.pos, 1)) and unit.unit_type == "HEAVY":
                    #すごく強くなるようならここを取る。
                    action = unit.dig(repeat=-1)
                    #print(f"{unit.unit_id} remote dig pow {unit.power}")
        if unit.unit_type == "HEAVY" and not is_on:
            #隣に敵がいる場合のHEAVYは最優先の生存で。敵とパワーを比較し、低ければ（または機関に必要なパワーギリギリなら）撤退を。
            enemy_units = g_state.units[enemy_team_id].values()
            target_unit = None
            for enemy_unit in enemy_units:
                if distance(enemy_unit.pos, unit.pos) == 1 and enemy_unit.unit_type == "HEAVY":
                    target_unit = enemy_unit
            if not(target_unit is None):
                #print("enemy_near!")
                return_action, cost = log_calc(g_state, unit, unit_log[unit.unit_id][0])
                if target_unit.power < unit.power and cost + unit.unit_cfg.MOVE_COST * 5 < unit.power and not in_terr:
                    #print(f"{unit.unit_id} offend")
                    action = unit.move(direction_to(unit.pos, target_unit.pos), repeat=-1)
                else:
                    #print(f"{unit.unit_id} deffend")
                    action = return_action

        return action

    def water_adj_pos(g_state:GameState):
        adj_vecs = np.array([[2,0],[2,1],[2,-1],[-2,0],[-2,1],[-2,-1],[0,2],[1,2],[-1,2],[0,-2],[1,-2],[-1,-2],\
            [3,0],[3,1],[3,-1],[-3,0],[-3,1],[-3,-1],[0,3],[1,3],[-1,3],[0,-3],[1,-3],[-1,-3],[2,2],[2,-2],[-2,2],[-2,-2]])
        ice_grids = []
        env_cfg = g_state.env_cfg
        for i in range(0,env_cfg.map_size):
            for j in range(0,env_cfg.map_size):
                if resoure_exist(g_state, np.array([i,j]), 0) == 1 :
                    ice_grids.append([i,j])
        return_grids = []
        for grid in ice_grids:
            for vec in adj_vecs:
                target_grid = [grid[0]+vec[0], grid[1]+vec[1]]
                if not(target_grid in return_grids) and not(pos_overrap_factory(g_state, np.array(target_grid)))\
                    and not pos_out_map(target_grid, g_state.env_cfg.map_size):
                    return_grids.append(target_grid)
        return np.array(return_grids)

    tokens = tokens.squeeze(0)
    actions = {}
    if(state.real_env_steps >= 0):
        #必要要素の計算
        my_lichen, opp_lichen, factory_pos_dict = get_pseudo_lichen_dict(state, agent)
        dig_pos, secondary_pos = get_tactical_points(state, my_lichen, opp_lichen, my_factories)
        total_tactical_pos = secondary_pos
        total_tactical_pos.extend(dig_pos)
        chase_pos = get_chase_points(state, agent)

        for index, factory in enumerate(my_factories.values()):
            action = rulebased_factory(state, factory, factory_pos_dict)
            if not action == None:
                actions[factory.unit_id] = action
                continue

        for index, unit in enumerate(my_units.values()):
            if(unit.power < unit.action_queue_cost(state)):
                continue
            
            action = rulebased_unit(state, unit, total_tactical_pos, factory_pos_dict, chase_pos)
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
                        action = unit.move(direction, repeat=-1)
                elif action_value < 2:#右折
                    direction = unit_log[unit.unit_id][1] % 4 + 1
                    cost = unit.move_cost(state, direction)
                    if not(cost == None) and unit.power >= cost:
                        action = unit.move(direction, repeat=-1)
                elif action_value < 3:#左折
                    direction = (unit_log[unit.unit_id][1] + 2) % 4 + 1
                    cost = unit.move_cost(state, direction)
                    if not(cost == None) and unit.power >= cost:
                        action = unit.move(direction, repeat=-1)
                elif action_value < 4:#transferはrule_based
                    direction_dict = {1:[0,-1], 2:[1, 0], 3:[0,1], 4:[-1,0]}
                    if unit.cargo.ice > 0 or unit.cargo.ore > 0:
                        is_adj, factory = unit_adjascent_factory(state, unit)
                        if is_adj:
                            direction = direction_to(unit.pos, factory.pos)
                            resource_type = 0 if unit.cargo.ice > 0 else 1
                            amount = unit.cargo.ice if resource_type == 0 else unit.cargo.ore
                            action = unit.transfer(direction, resource_type, amount, repeat=0)
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
                                action = unit.transfer(direction, resource_type, amount, repeat=0)
                                #if unit.cargo.ice or unit.cargo.ore:
                                    #print(f"{unit.unit_id} tarnser to {target_unit.unit_id}, {resource_type} {amount}")
                    else:
                        direction = unit_log[unit.unit_id][1]
                        cost = unit.move_cost(state, direction)
                        if not(cost == None) and unit.power >= cost:
                            action = unit.move(direction, repeat=-1)
                elif action_value < 5:#pick_upはiceかoreのみ
                    if unit_on_factory(state, unit)[0]:
                        resource_type = int(((action_value-2) * 2) % 2)
                        action = unit.pickup(resource_type, 100, repeat=-1)
                    elif unit.power >= unit.dig_cost(state):
                        action = unit.dig(repeat=-1)
                elif action_value < 6:#digだが、自身破壊入れるならここ
                    is_on, factory_ = unit_on_factory(state, unit)
                    if is_on:
                        action = unit.pickup(4, min(unit.dig_cost(state)*5, factory_.power), repeat=-1)
                    elif unit.power >= unit.dig_cost(state):
                        direction = unit_log[unit.unit_id][1]
                        cost = unit.move_cost(state, direction)
                        if rubble_num(state, unit.pos):
                            action = unit.dig(repeat=-1)
                        elif not(cost == None) and unit.power >= unit.move_cost(state, direction):
                            action = unit.move(direction, repeat=-1)
                else:
                    print("error-tipo")
            
            #print(f"`{unit.unit_id}, {action}, {unit_next_action(unit)}")
            if not (action is None) and not is_the_same_action(action, unit_next_action(unit)):
                #print("change")
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
        water_adjs:np.ndarray = water_adj_pos(state)
        potential_spawns:np.ndarray = state.board.valid_spawns_mask
        normal_potentials = []
        for i in range(potential_spawns.shape[0]):
            for j in range(potential_spawns.shape[1]):
                if potential_spawns[i][j]:
                    normal_potentials.append(np.array([i,j]))
        water_potentials = []
        for i in range(water_adjs.shape[0]):
            for j in range(len(normal_potentials)):
                if distance(water_adjs[i], normal_potentials[j]) == 0:
                    water_potentials.append(water_adjs[i])
        #print(water_potentials)
        length = 100
        index = 0
        if len(water_potentials) > 0:
            for i in range(len(water_potentials)):
                if pos_overrap_factory(state, water_potentials[i]):
                    continue
                length_ = abs(water_potentials[i][0]-pos[0])+abs(water_potentials[i][1]-pos[1])
                if length_ < length:
                    index = i
                    length = length_
            actions = dict(spawn = water_potentials[index], metal = 150, water = 150)
        else:
            for i in range(len(normal_potentials)):
                if pos_overrap_factory(state, normal_potentials[i]):
                    continue
                length_ = abs(normal_potentials[i][0]-pos[0])+abs(normal_potentials[i][1]-pos[1])
                if length_ < length:
                    index = i
                    length = length_
            actions = dict(spawn = normal_potentials[index], metal = 150, water = 150)
    else:
        actions = dict(faction="AlphaStrike", bid = 0)

    #print(f"actions = {actions}")
    return actions

def env_to_tokens(state:GameState, unit_log, view_agent):#雑に作る。若干の情報のオーバーラップは仕方なし。
    board = state.board
    env_cfg = state.env_cfg
    agents = []
    if view_agent == "player_0":
        agents = ["player_0", "player_1"]
    elif view_agent == "player_1":
        agents = ["player_1", "player_0"]
    else:
        print("error tipo")
    def rubble_lichen_magnitude(amount:int) -> int:
        if amount == 0:
            return 0
        elif 0 < amount and amount <= 100:
            return 50 + amount/2
        else:
            print("rl_magnitude_out of length")
            return 100
    #まずはrubble(*2)
    rubble_map = board.rubble
    rubble_map_informal = np.zeros((env_cfg.map_size//2, env_cfg.map_size//2))
    for i in range(env_cfg.map_size//2):
        for j in range(env_cfg.map_size//2):
            for k in range(2):
                for l in range(2):
                    rubble_map_informal[i][j] = rubble_lichen_magnitude(rubble_map[2*i+k][2*j+l])/400
    rubble_map_informal = rubble_map_informal.reshape((rubble_map_informal.size // token_len, token_len))
    tokens = rubble_map_informal
    #次にresources(*8)
    ice_map = board.ice
    ore_map = board.ore
    resource_map = ice_map + 2 * ore_map
    resource_map = resource_map.reshape((resource_map.size // token_len, token_len))
    tokens = np.concatenate((tokens, resource_map))
    #次にlichen(*2)
    lichen_map = np.zeros((env_cfg.map_size//2, env_cfg.map_size//2))
    my_lichen_dict, opp_lichen_dict = get_lichen_dict(state, view_agent)
    for i in range(env_cfg.map_size//2):
        for j in range(env_cfg.map_size//2):
            for k in range(2):
                for l in range(2):
                    grid = np.array([2*i+k, 2*j+l]).astype(np.int32)
                    grid_byte = grid.tobytes()
                    if grid_byte in my_lichen_dict:
                        lichen_map[i][j] += rubble_lichen_magnitude(my_lichen_dict[grid_byte])/400
    lichen_map = lichen_map.reshape((lichen_map.size // token_len, token_len))
    tokens = np.concatenate((tokens, lichen_map))
    #次にfactory(場所:2, リソース:5, strain_id:1, player:1)最高でも5個らしい(*1)
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
    #次にunit(場所:2, リソース:5, playerと種別:1, 次の行動:1)(*2)
    unit_info_dim = 9
    unit_num = 0
    for agent in state.units:
        unit_num += len(state.units[agent].values())
    #unit_infos = np.zeros((unit_num//(token_len//unit_info_dim) + 1, token_len))
    unit_infos = np.zeros((2, token_len))
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
                if x_index < 1:
                    x_index += 1
    tokens = np.concatenate((tokens, unit_infos))
    #ラスト基本情報(*1)
    basics = np.zeros((1, token_len))
    basics[0][0] = state.real_env_steps / state.env_cfg.max_episode_length
    if state.real_env_steps < 0:
        pass
    elif state.env_cfg.max_episode_length - state.real_env_steps > token_len - 1:
        #print(state.weather_schedule[state.real_env_steps:state.real_env_steps+token_len-1])
        basics[0][1:] = state.weather_schedule[state.real_env_steps:state.real_env_steps+token_len-1]
    elif state.real_env_steps < state.env_cfg.max_episode_length:
        #print(state.weather_schedule[state.real_env_steps:])
        basics[0][1:1+state.env_cfg.max_episode_length - state.real_env_steps] = state.weather_schedule[state.real_env_steps:]
    tokens = np.concatenate((tokens, basics))
    
    #print(tokens.shape)
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
        adj_grids = factory_adj_grids(factory, state.env_cfg)
        for grid in adj_grids:
            if resoure_exist(state, grid, 0):
                value += 0.05
                break
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
        if unit.power > unit.dig_cost(state) * 2 and ((unit_id in action.keys() and (all(action[unit_id][0] == unit.dig(repeat=-1)) or all(action[unit_id][0] == unit.dig(repeat=-1)))) or\
            (not(unit_id in action.keys()) and len(unit.action_queue) > 0 and (all(unit.action_queue[0] == unit.dig(repeat=-1)) or all(unit.action_queue[0] == unit.dig(repeat=-1))))):
            pos = unit.pos
            if resoure_exist(state, pos, 0):
                value += 0.01
                #print("digging ice")
            elif resoure_exist(state, pos, 1):
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

def Play(v_net: ValueNet, a_net: ActionNet, d_net:CustomNet, s_net:CustomNet, beta, create_log = False):

    def log_addition(log:Tuple, pos:np.ndarray):
        def check_is_inlog():
            exist = False
            index = -1
            for i in range(len(log[0])):
                if distance(pos, log[0][i]) == 0:
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

    def check_is_finished(state:GameState, game_len):
        finished = False
        cause:str = ""
        factories = state.factories.items()
        for agent, factory in factories:
            if len(factory) == 0 and state.real_env_steps > 0:
                finished = True
                cause += f"[{agent} down] "
        if state.real_env_steps == game_len:
            finished = True
            agents = ["player_0", "player_1"]
            cause += f"[env leached last, value(0vs1): {state_value(state, agents[0])}: {state_value(state, agents[1])}]"
        return finished, cause
    env = LuxAI2022(verbose = 0)
    seed = random.randint(0, 100000)
    step = 0
    env_cfg = env.env_cfg
    obs = env.reset(seed = seed)["player_0"]
    state = obs_to_game_state(step, env_cfg, obs)
    unit_log = {}
    state_tokens_0 = env_to_tokens(state, unit_log, "player_0")
    state_tokens_1 = env_to_tokens(state, unit_log, "player_1")
    state_value_0 = state_value(state, "player_0")
    state_value_1 = state_value(state, "player_1")
    states_0 = [state_tokens_0]
    actions_0 = []
    returns_0 = []
    states_1 = [state_tokens_1]
    actions_1 = []
    returns_1 = []
    state_obs = env.state.get_compressed_obs()
    replay = {"observations": [state_obs], "actions":[{}]}
    while not check_is_finished(state, env_cfg.max_episode_length)[0]:#環境が終わるまで
        state_tokens_0 = torch.from_numpy(env_to_tokens(state, unit_log, "player_0")).float().to(device)
        state_tokens_1 = torch.from_numpy(env_to_tokens(state, unit_log, "player_1")).float().to(device)
        a_0_t = a_net(state_tokens_0)
        a_1_t = a_net(state_tokens_1)
        a_0 = a_0_t.to('cpu').detach().numpy()
        a_1 = a_1_t.to('cpu').detach().numpy()
        value_0 = v_net(torch.concat([state_tokens_0, a_0_t])).item()
        default_0:np.ndarray = d_net(torch.concat([state_tokens_0, a_0_t]).unsqueeze(0)).to('cpu').detach().numpy()
        search_0:np.ndarray = s_net(torch.concat([state_tokens_0, a_0_t]).unsqueeze(0)).to('cpu').detach().numpy()
        search_mse_0 = ((default_0 - search_0) ** 2).mean()
        value_1 = v_net(torch.concat([state_tokens_1, a_1_t])).item()
        default_1:np.ndarray = d_net(torch.concat([state_tokens_1, a_1_t]).unsqueeze(0)).to('cpu').detach().numpy()
        search_1:np.ndarray = s_net(torch.concat([state_tokens_1, a_1_t]).unsqueeze(0)).to('cpu').detach().numpy()
        search_mse_1 = ((default_1 - search_1) ** 2).mean()
        for i in range(10):#もちろん時間で区切ってもよし
            a_0_ = action_nearby_token(a_0, beta)
            a_0_t_ = torch.from_numpy(a_0_).float().to(device)
            value_0_ = v_net(torch.concat([state_tokens_0, a_0_t_])).item()
            default_0_:np.ndarray = d_net(torch.concat([state_tokens_0, a_0_t_]).unsqueeze(0)).to('cpu').detach().numpy()
            search_0_:np.ndarray = s_net(torch.concat([state_tokens_0, a_0_t_]).unsqueeze(0)).to('cpu').detach().numpy()
            search_mse_0_ = ((default_0_ - search_0_) ** 2).mean()
            a_1_ = action_nearby_token(a_1, beta)
            a_1_t_ = torch.from_numpy(a_1_).float().to(device)
            value_1_ = v_net(torch.concat([state_tokens_1, a_1_t_])).item()
            default_1_:np.ndarray = d_net(torch.concat([state_tokens_1, a_1_t_]).unsqueeze(0)).to('cpu').detach().numpy()
            search_1_:np.ndarray = s_net(torch.concat([state_tokens_1, a_1_t_]).unsqueeze(0)).to('cpu').detach().numpy()
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
        real_actions_0 = tokens_to_actions(state, a_0, "player_0", unit_log)
        action_value_0 = action_value(state, real_actions_0, "player_0")
        real_actions_1 = tokens_to_actions(state, a_1, "player_1", unit_log)
        action_value_1 = action_value(state, real_actions_1, "player_1")
        real_actions = {"player_0":real_actions_0, "player_1":real_actions_1}
        obs = env.step(real_actions)[0]["player_0"]
        step += 1
        state = obs_to_game_state(step, env_cfg, obs)
        #log更新はstate確定直後に
        units_0 = state.units["player_0"]
        for unit in units_0.values():
            is_on, factory = unit_on_factory(state, unit)
            if not(unit.unit_id in unit_log.keys()):
                unit_log[unit.unit_id] = ([unit.pos], random.randint(1, 4))
            elif is_on:
                direction = direction_to(factory.pos, unit.pos)
                if direction == 0:
                    direction = random.randint(1, 4)
                unit_log[unit.unit_id] = ([unit.pos], direction)
            else:
                unit_log[unit.unit_id] = log_addition(unit_log[unit.unit_id], unit.pos)
        units_1 = state.units["player_1"]
        for unit in units_1.values():
            is_on, factory = unit_on_factory(state, unit)
            if not(unit.unit_id in unit_log.keys()):
                unit_log[unit.unit_id] = ([unit.pos], random.randint(1, 4))
            elif is_on:
                direction = direction_to(factory.pos, unit.pos)
                if direction == 0:
                    direction = random.randint(1, 4)
                unit_log[unit.unit_id] = ([unit.pos], direction)
            else:
                unit_log[unit.unit_id] = log_addition(unit_log[unit.unit_id], unit.pos)
        state_tokens_0 = env_to_tokens(state, unit_log, "player_0")
        state_tokens_1 = env_to_tokens(state, unit_log, "player_1")
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
        if create_log:
            replay["observations"].append(env.state.get_change_obs(state_obs))
            state_obs = obs
            replay["actions"].append(real_actions)          
    print(f"env {seed} finished in step {state.real_env_steps} cause = {check_is_finished(state, env_cfg.max_episode_length)[1]}")
    if create_log:
        with open("./battle_log.json", "w") as outfile: 
            json.dump(to_json(replay), outfile)
            print("saved log")


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
                (beta_max * variance), results_num < 100)
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
        print(f"ver1.15.3 restart from epoch {restart_epoch}")
        Train()
    elif arg[1] == "__predict":
        #実行の挙動を定義
        pass