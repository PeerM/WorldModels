# xvfb-run -s "-screen 0 1400x900x24" python generate_data.py car-racing --total_episodes 200 --start_batch 0 --time_steps 300

import numpy as np
import random
import config
# import matplotlib.pyplot as plt

import hsa.post.nes_env

import argparse

from extern.fceux_learningenv.nes_python_interface import NESInterface
from hsa.nes_python_input import py_to_nes_wrapper
from hsa.visualization.parse_fm2 import parse_fm2


def main(args):
    start_batch = args.start_batch
    render = args.render

    inputs = []
    with open(args.movie) as movie_file:
        py_inputs = list(parse_fm2(movie_file))
        inputs = list(map(py_to_nes_wrapper, py_inputs))

    nes_low = NESInterface("/home/peer/mario.nes", eb_compatible=False, auto_render_period=1)
    env = hsa.post.nes_env.NesEnv(frameskip=1, obs_type="image", nes=nes_low, raw_actions=True)

    obs_data = []
    action_data = []

    print('-----')
    observation = None
    last_action = None
    # observation, reward, done, info = env.step(0)
    # observation = config.adjust_obs(observation, args.scale)

    env.render()
    obs_sequence = []
    action_sequence = []

    for current_action, py_action in zip(inputs, py_inputs):

        if observation is not None:
            obs_sequence.append(observation)
            action_sequence.append(last_action)

        observation, reward, done, info = env.step(current_action)
        observation = config.adjust_obs(observation, args.scale)

        if render:
            env.render()
        last_action = current_action

    obs_data = obs_sequence
    action_data = action_sequence

    print("Current dataset contains {} observations".format(sum(map(len, obs_data))))

    print("Saving dataset for batch {}".format(args.movie_name))
    np.save('./data/obs_data_' + args.movie_name, obs_data)
    np.save('./data/action_data_' + args.movie_name, action_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Create new training data'))
    parser.add_argument('movie', type=str)
    parser.add_argument('movie_name', type=str)
    parser.add_argument('--start_batch', type=int, default=0, help='start_batch number')
    parser.add_argument('--render', action='store_true', help='render the env as data is generated')
    parser.add_argument('--scale', action="store_true", help="should the frames be scaled down?")

    args = parser.parse_args()
    main(args)

    config
