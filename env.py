import numpy as np
# import gym
import hsa.gen3.nes_env
from custom_envs.car_racing import CarRacing
from extern.fceux_learningenv.nes_python_interface import NESInterface


def make_env(env_name, seed=-1, render_mode=False):
    if env_name == 'car_racing':
        env = CarRacing()
        if (seed >= 0):
            env.seed(seed)
    if env_name == 'mario':
        nes_low = NESInterface("/home/peer/mario.nes", eb_compatible=False, auto_render_period=1)
        env = hsa.gen3.nes_env.NesEnv(frameskip=4, obs_type="image", nes=nes_low)
    else:
        print("couldn't find this env")

    return env
