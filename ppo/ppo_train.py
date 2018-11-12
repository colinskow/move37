# Based on https://github.com/higgsfield/RL-Adventure-2/blob/master/3.ppo.ipynb
import argparse
import math
import os
import random
import gym
import numpy as np

# Follow instructions here to install https://github.com/openai/roboschool
import roboschool

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from tensorboardX import SummaryWriter

from lib.common import mkdir
from lib.model import ActorCritic
from lib.multiprocessing_env import SubprocVecEnv

NUM_ENVS            = 8 
ENV_ID              = "RoboschoolHalfCheetah-v1"
HIDDEN_SIZE         = 256
LEARNING_RATE       = 1e-4
GAMMA               = 0.99
GAE_LAMBDA          = 0.95
PPO_EPSILON         = 0.2
CRITIC_DISCOUNT     = 0.5
ENTROPY_BETA        = 0.001
PPO_STEPS           = 256
MINI_BATCH_SIZE     = 64
PPO_EPOCHS          = 10
TEST_EPOCHS         = 10
NUM_TESTS           = 10
TARGET_REWARD       = 2500


def make_env():
    # returns a function which creates a single environment
    def _thunk():
        env = gym.make(ENV_ID)
        return env
    return _thunk

    
def test_env(env, model, device, deterministic=True):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = model(state)
        action = dist.mean.detach().cpu().numpy()[0] if deterministic \
            else dist.sample().cpu().numpy()[0]
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
    return total_reward


def normalize(x):
    x -= x.mean()
    x /= (x.std() + 1e-8)
    return x


def compute_gae(next_value, rewards, masks, values, gamma=GAMMA, lam=GAE_LAMBDA):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        # prepend to get correct order back
        returns.insert(0, gae + values[step])
    return returns


def ppo_iter(states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    # generates random mini-batches until we have covered the full batch
    for _ in range(batch_size // MINI_BATCH_SIZE):
        rand_ids = np.random.randint(0, batch_size, MINI_BATCH_SIZE)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]
        

def ppo_update(frame_idx, states, actions, log_probs, returns, advantages, clip_param=PPO_EPSILON):
    count_steps = 0
    sum_returns = 0.0
    sum_advantage = 0.0
    sum_loss_actor = 0.0
    sum_loss_critic = 0.0
    sum_entropy = 0.0
    sum_loss_total = 0.0

    # PPO EPOCHS is the number of times we will go through ALL the training data to make updates
    for _ in range(PPO_EPOCHS):
        # grabs random mini-batches several times until we have covered all data
        for state, action, old_log_probs, return_, advantage in ppo_iter(states, actions, log_probs, returns, advantages):
            dist, value = model(state)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = CRITIC_DISCOUNT * critic_loss + actor_loss - ENTROPY_BETA * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # track statistics
            sum_returns += return_.mean()
            sum_advantage += advantage.mean()
            sum_loss_actor += actor_loss
            sum_loss_critic += critic_loss
            sum_loss_total += loss
            sum_entropy += entropy
            
            count_steps += 1
    
    writer.add_scalar("returns", sum_returns / count_steps, frame_idx)
    writer.add_scalar("advantage", sum_advantage / count_steps, frame_idx)
    writer.add_scalar("loss_actor", sum_loss_actor / count_steps, frame_idx)
    writer.add_scalar("loss_critic", sum_loss_critic / count_steps, frame_idx)
    writer.add_scalar("entropy", sum_entropy / count_steps, frame_idx)
    writer.add_scalar("loss_total", sum_loss_total / count_steps, frame_idx)


if __name__ == "__main__":
    mkdir('.', 'checkpoints')
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", default=ENV_ID, help="Name of the run")
    args = parser.parse_args()
    writer = SummaryWriter(comment="ppo_" + args.name)
    
    # Autodetect CUDA
    use_cuda = torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")
    print('Device:', device)
    
    # Prepare environments
    envs = [make_env() for i in range(NUM_ENVS)]
    envs = SubprocVecEnv(envs)
    env = gym.make(ENV_ID)
    num_inputs  = envs.observation_space.shape[0]
    num_outputs = envs.action_space.shape[0]

    model = ActorCritic(num_inputs, num_outputs, HIDDEN_SIZE).to(device)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    frame_idx  = 0
    train_epoch = 0
    best_reward = None

    state = envs.reset()
    early_stop = False

    while not early_stop:

        log_probs = []
        values    = []
        states    = []
        actions   = []
        rewards   = []
        masks     = []

        for _ in range(PPO_STEPS):
            state = torch.FloatTensor(state).to(device)
            dist, value = model(state)

            action = dist.sample()
            # each state, reward, done is a list of results from each parallel environment
            next_state, reward, done, _ = envs.step(action.cpu().numpy())
            log_prob = dist.log_prob(action)
            
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
            masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
            
            states.append(state)
            actions.append(action)
            
            state = next_state
            frame_idx += 1
                
        next_state = torch.FloatTensor(next_state).to(device)
        _, next_value = model(next_state)
        returns = compute_gae(next_value, rewards, masks, values)

        returns   = torch.cat(returns).detach()
        log_probs = torch.cat(log_probs).detach()
        values    = torch.cat(values).detach()
        states    = torch.cat(states)
        actions   = torch.cat(actions)
        advantage = returns - values
        advantage = normalize(advantage)
        
        ppo_update(frame_idx, states, actions, log_probs, returns, advantage)
        train_epoch += 1

        if train_epoch % TEST_EPOCHS == 0:
            test_reward = np.mean([test_env(env, model, device) for _ in range(NUM_TESTS)])
            writer.add_scalar("test_rewards", test_reward, frame_idx)
            print('Frame %s. reward: %s' % (frame_idx, test_reward))
            # Save a checkpoint every time we achieve a best reward
            if best_reward is None or best_reward < test_reward:
                if best_reward is not None:
                    print("Best reward updated: %.3f -> %.3f" % (best_reward, test_reward))
                    name = "%s_best_%+.3f_%d.dat" % (args.name, test_reward, frame_idx)
                    fname = os.path.join('.', 'checkpoints', name)
                    torch.save(model.state_dict(), fname)
                best_reward = test_reward
            if test_reward > TARGET_REWARD: early_stop = True
