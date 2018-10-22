# Forked from https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On

import gym
import time
import argparse
import numpy as np

import ptan
import torch

from configs import PARAMS
from gym import wrappers
from lib.doom_wrappers import make_doom_env
from lib import model
from lib.common import mkdir

import collections

FPS = 25


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default='pong', help="Config to use")
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-r", "--record", help="Directory to store video recording")
    parser.add_argument("--no-visualize", default=True, action='store_false', dest='visualize',
                        help="Disable visualization of the game play")
    args = parser.parse_args()
    params = PARAMS[args.config]

    if args.config == 'doom':
        import vizdoomgym
        env = make_doom_env(params['env_name'])
    else:
        env = gym.make(params['env_name'])
        env = ptan.common.wrappers.wrap_dqn(env)
    
    if args.record:
        mkdir('.', args.record)
        env = wrappers.Monitor(env, args.record, force=True)
    net = model.NoisyDuelingDQN(env.observation_space.shape, env.action_space.n)
    net.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))

    state = env.reset()
    total_reward = 0.0
    c = collections.Counter()

    while True:
        start_ts = time.time()
        if args.visualize:
            env.render()
        state_v = torch.tensor(np.expand_dims(np.array(state, copy=False), axis=0))
        q_vals = net(state_v).data.numpy()[0]
        action = np.argmax(q_vals)
        c[action] += 1
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
        
        if args.visualize:
            delta = 1/FPS - (time.time() - start_ts)
            if delta > 0:
                time.sleep(delta)
        
    print("Total reward: %.2f" % total_reward)
    print("Action counts:", c)
    env.env.close()
