import numpy as np
# direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
def direction_to(src, target):
    if type(src) == list:
        src = np.array(src)
    if type(target) == list:
        target = np.array(target)
    ds = target - src
    dx = ds[0]
    dy = ds[1]
    if dx == 0 and dy == 0:
        return 0
    if abs(dx) > abs(dy):
        if dx > 0:
            return 2 
        else:
            return 4
    elif abs(dx) < abs(dy):
        if dy > 0:
            return 3
        else:
            return 1
    else:
        if dx > 0 and dy > 0:
            return 3
        elif dx > 0 and dy < 0:
            return 2
        elif dx < 0 and dy < 0:
            return 1
        elif dx < 0 and dy > 0:
            return 4
        else:
            print("error in utils")

def secondery_directiopn_to(src, target):
    if type(src) == list:
        src = np.array(src)
    if type(target) == list:
        target = np.array(target)
    ds = target - src
    dx = ds[0]
    dy = ds[1]
    if dx == 0 and dy == 0:
        return 0
    if abs(dx) > abs(dy):
        if dx > 0:
            return 3 
        else:
            return 1
    elif abs(dx) < abs(dy):
        if dy > 0:
            return 2
        else:
            return 4
    else:
        if dx > 0 and dy > 0:
            return 2
        elif dx > 0 and dy < 0:
            return 1
        elif dx < 0 and dy < 0:
            return 4
        elif dx < 0 and dy > 0:
            return 3
        else:
            print("error in utils")

def is_the_same_action(src:np.ndarray, tgt:np.ndarray):
    if not src[0] == tgt[0]:
        return False
    else:
        if src[0] == 0:
            tgt[2] = 0
            src[2] = 0
        return all(src == tgt) 

def move_effective_direction_to(src, target):
    if type(src) == list:
        src = np.array(src)
    if type(target) == list:
        target = np.array(target)
    ds = target - src
    dx = ds[0]
    dy = ds[1]
    if dx == 0 and dy == 0:
        return 0
    if abs(dx) > 0:
        if dx > 0:
            return 2 
        else:
            return 4
    else:
        if dy > 0:
            return 3
        else:
            return 1

def secondary_move_effective_direction_to(src, target):
    if type(src) == list:
        src = np.array(src)
    if type(target) == list:
        target = np.array(target)
    ds = target - src
    dx = ds[0]
    dy = ds[1]
    if dx == 0 and dy == 0:
        return 0
    if abs(dx) > 0:
        if dy > 0:
            return 3
        else:
            return 1
    else:
        if dy > 0:
            return 2
        else:
            return 4