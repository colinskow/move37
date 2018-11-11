import argparse
import gym
import roboschool

from lib.model import ActorCritic

import numpy as np
import torch


ENV_ID = "RoboschoolHalfCheetah-v1"
HIDDEN_SIZE = 256

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-e", "--env", default=ENV_ID, help="Environment name to use, default=" + ENV_ID)
    parser.add_argument("-d", "--deterministic", default=False, action="store_true", help="enable deterministic actions")
    parser.add_argument("-r", "--record", help="If specified, sets the recording dir, default=Disabled")
    args = parser.parse_args()

    # Autodetect CUDA
    use_cuda = torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")

    env = gym.make(args.env)
    if args.record:
        env = gym.wrappers.Monitor(env, args.record, force=True)

    num_inputs  = env.observation_space.shape[0]
    num_outputs = env.action_space.shape[0]
    model = ActorCritic(num_inputs, num_outputs, HIDDEN_SIZE).to(device)
    model.load_state_dict(torch.load(args.model))

    state = env.reset()
    done = False
    total_steps = 0
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = model(state)
        action = dist.mean.detach().cpu().numpy()[0] if args.deterministic \
            else dist.sample().cpu().numpy()[0]
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
        total_steps += 1
    env.env.close()
    print("In %d steps we got %.3f reward" % (total_steps, total_reward))
