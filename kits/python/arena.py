from lux.kit import obs_to_game_state, GameState, EnvConfig
from lux.utils import direction_to
import numpy as np
import sys
import torch
import torch.nn as nn
from lux.kit import Team, Factory, Unit, UnitCargo
import math
import os
from agent import Agent
import matplotlib.pyplot as plt
from luxai2022.utils import animate
from luxai2022 import LuxAI2022
import random

def interact(env, agents, steps):
    # reset our env
    seed = random.randint(0, 10000)
    print(f"env seed = {seed}")
    obs = env.reset(seed)
    np.random.seed(0)
    imgs = []
    step = 0
    # Note that as the environment has two phases, we also keep track a value called 
    # `real_env_steps` in the environment state. The first phase ends once `real_env_steps` is 0 and used below

    # iterate until phase 1 ends
    while env.state.real_env_steps < 0:
        if step >= steps: break
        actions = {}
        for player in env.agents:
            o = obs[player]
            a = agents[player].early_setup(step, o)
            actions[player] = a
        step += 1
        obs, rewards, dones, infos = env.step(actions)
        imgs += [env.render("rgb_array")]
    done = False
    while not done:
        if step >= steps: break
        actions = {}
        for player in env.agents:
            o = obs[player]
            a = agents[player].act(step, o)
            actions[player] = a
        step += 1
        obs, rewards, dones, infos = env.step(actions)
        imgs += [env.render("rgb_array")]
        done = dones["player_0"] and dones["player_1"]
    return animate(imgs)

if __name__ == "__main__":
    env = LuxAI2022()
    env.reset(seed = 42)
    agents = {player: Agent(player, env.state.env_cfg) for player in env.agents}
    interact(env, agents, 1000)