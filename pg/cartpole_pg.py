# Forked from https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On

import gym
import numpy as np
import argparse
import collections

# https://github.com/Shmuma/ptan
import ptan
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PGN(nn.Module):
    def __init__(self, input_size, n_actions):
        super(PGN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)


class MeanBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.deque = collections.deque(maxlen=capacity)
        self.sum = 0.0

    def add(self, val):
        if len(self.deque) == self.capacity:
            self.sum -= self.deque[0]
        self.deque.append(val)
        self.sum += val

    def mean(self):
        if not self.deque:
            return 0.0
        return self.sum / len(self.deque)


TARGET_REWARD = 195
GAMMA = 0.99
LEARNING_RATE = 0.001
BATCH_SIZE = 32
ENTROPY_BETA = 0.01

BELLMAN_STEPS = 10
BASELINE_STEPS = 50000


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = gym.make("CartPole-v0")
    writer = SummaryWriter(comment="-cartpole-pg")

    net = PGN(env.observation_space.shape[0], env.action_space.n).to(device)
    print(net)

    # The agent inputs the state into the neural network, and gets back logits.
    # It puts these through a softmax to get probabilities, then samples an action.
    agent = ptan.agent.PolicyAgent(net, preprocessor=ptan.agent.float32_preprocessor,
                                   apply_softmax=True, device=device)
    # The experience source interacts with the environment and returns (s,a,r,s') transitions
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=BELLMAN_STEPS)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    total_rewards = []
    step_rewards = []
    baseline_buf = MeanBuffer(BASELINE_STEPS)
    step_idx = 0
    done_episodes = 0

    batch_states, batch_actions, batch_scales = [], [], []

    # each iteration runs one action in the environment and returns a (s,a,r,s') transition
    for step_idx, exp in enumerate(exp_source):
        baseline_buf.add(exp.reward)
        baseline = baseline_buf.mean()
        writer.add_scalar("baseline", baseline, step_idx)
        batch_states.append(exp.state)
        batch_actions.append(int(exp.action))
        batch_scales.append(exp.reward - baseline)

        # handle when an episode is completed
        episode_rewards = exp_source.pop_total_rewards()
        if episode_rewards:
            done_episodes += 1
            reward = episode_rewards[0]
            total_rewards.append(reward)
            mean_rewards = float(np.mean(total_rewards[-100:]))
            print("%d: reward: %6.2f, mean_100: %6.2f, episodes: %d" % (
                step_idx, reward, mean_rewards, done_episodes))
            writer.add_scalar("reward", reward, step_idx)
            writer.add_scalar("reward_100", mean_rewards, step_idx)
            writer.add_scalar("episodes", done_episodes, step_idx)
            if mean_rewards > TARGET_REWARD:
                print("Solved in %d steps and %d episodes!" % (step_idx, done_episodes))
                break

        if len(batch_states) < BATCH_SIZE:
            continue

        # copy training data to the GPU
        states_v = torch.FloatTensor(batch_states).to(device)
        batch_actions_t = torch.LongTensor(batch_actions).to(device)
        batch_scale_v = torch.FloatTensor(batch_scales).to(device)

        # apply gradient descent
        optimizer.zero_grad()
        logits_v = net(states_v)
        # apply the softmax and take the logarithm in one step, more precise
        log_prob_v = F.log_softmax(logits_v, dim=1)
        # scale the log probs according to (reward - baseline)
        log_prob_actions_v = batch_scale_v * log_prob_v[range(BATCH_SIZE), batch_actions_t]
        # take the mean cross-entropy across all batches
        loss_policy_v = -log_prob_actions_v.mean()

        # subtract the entropy bonus from the loss function
        prob_v = F.softmax(logits_v, dim=1)
        entropy_v = -(prob_v * log_prob_v).sum(dim=1).mean()
        entropy_loss_v = -ENTROPY_BETA * entropy_v
        loss_v = loss_policy_v + entropy_loss_v

        loss_v.backward()
        optimizer.step()

        # calc KL-divergence, for graphing puproses only
        new_logits_v = net(states_v)
        new_prob_v = F.softmax(new_logits_v, dim=1)
        kl_div_v = -((new_prob_v / prob_v).log() * prob_v).sum(dim=1).mean()
        writer.add_scalar("kl", kl_div_v.item(), step_idx)

        # track statistics on the gradients for Tensorboard
        grad_max = 0.0
        grad_means = 0.0
        grad_count = 0
        for p in net.parameters():
            grad_max = max(grad_max, p.grad.abs().max().item())
            grad_means += (p.grad ** 2).mean().sqrt().item()
            grad_count += 1

        writer.add_scalar("baseline", baseline, step_idx)
        writer.add_scalar("entropy", entropy_v.item(), step_idx)
        writer.add_scalar("batch_scales", np.mean(batch_scales), step_idx)
        writer.add_scalar("loss_entropy", entropy_loss_v.item(), step_idx)
        writer.add_scalar("loss_policy", loss_policy_v.item(), step_idx)
        writer.add_scalar("loss_total", loss_v.item(), step_idx)
        writer.add_scalar("grad_l2", grad_means / grad_count, step_idx)
        writer.add_scalar("grad_max", grad_max, step_idx)

        batch_states.clear()
        batch_actions.clear()
        batch_scales.clear()

    writer.close()
