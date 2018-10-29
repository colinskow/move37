# Forked from https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On

import math
import gym
import ptan
import numpy as np
import argparse
import collections
from tensorboardX import SummaryWriter

import torch
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

from lib import common

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 128

BELLMAN_STEPS = 4
CLIP_GRAD = 0.1

TOTAL_ENVS = 64
PROCESSES_COUNT = mp.cpu_count()
ENVS_PER_PROCESS = math.ceil(TOTAL_ENVS / PROCESSES_COUNT)

if True:
    ENV_NAME = "PongNoFrameskip-v4"
    NAME = 'pong'
    REWARD_BOUND = 18
else:
    ENV_NAME = "BreakoutNoFrameskip-v4"
    NAME = "breakout"
    REWARD_BOUND = 400


def make_env():
    return ptan.common.wrappers.wrap_dqn(gym.make(ENV_NAME))

TotalReward = collections.namedtuple('TotalReward', field_names='reward')


# data_func is forked and run by each process
# It grabs one experience transition from an environment and adds it to the training queue
def data_func(net, device, train_queue):
    # each process runs multiple instances of the environment, round-robin
    envs = [make_env() for _ in range(ENVS_PER_PROCESS)]
    agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], device=device, apply_softmax=True)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, gamma=GAMMA, steps_count=BELLMAN_STEPS)

    for exp in exp_source:
        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            train_queue.put(TotalReward(reward=np.mean(new_rewards)))
        train_queue.put(exp)


if __name__ == "__main__":
    common.mkdir('.', 'checkpoints')
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()
    device = "cuda" if args.cuda else "cpu"

    writer = SummaryWriter(comment="-a3c-data_" + NAME + "_" + args.name)

    env = make_env()
    net = common.AtariA2C(env.observation_space.shape, env.action_space.n).to(device)
    net.share_memory()
    print(net)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)

    train_queue = mp.Queue(maxsize=PROCESSES_COUNT)
    data_proc_list = []
    # Spawn processes to run data_func
    for _ in range(PROCESSES_COUNT):
        data_proc = mp.Process(target=data_func, args=(net, device, train_queue))
        data_proc.start()
        data_proc_list.append(data_proc)

    batch = []
    step_idx = 0

    try:
        with common.RewardTracker(writer, stop_reward=REWARD_BOUND) as tracker:
            with ptan.common.utils.TBMeanTracker(writer, batch_size=100) as tb_tracker:
                while True:
                    # Get one transition from the training queue
                    train_entry = train_queue.get()
                    # If the episode is over we will receive the total reward from that episode
                    if isinstance(train_entry, TotalReward):
                        finished, save_checkpoint = tracker.reward(train_entry.reward, step_idx)
                        if save_checkpoint: 
                            torch.save(net.state_dict(), './checkpoints/' + args.name + "-best.dat")
                        if finished:
                            break
                        continue

                    step_idx += 1
                    # keep receiving data until one batch is full
                    batch.append(train_entry)
                    if len(batch) < BATCH_SIZE:
                        continue

                    states_v, actions_t, q_vals_v = \
                        common.unpack_batch(batch, net, last_val_gamma=GAMMA**BELLMAN_STEPS, device=device)
                    batch.clear()

                    optimizer.zero_grad()
                    logits_v, value_v = net(states_v)

                    loss_value_v = F.mse_loss(value_v.squeeze(-1), q_vals_v)

                    log_prob_v = F.log_softmax(logits_v, dim=1)
                    adv_v = q_vals_v - value_v.detach()
                    log_prob_actions_v = adv_v * log_prob_v[range(BATCH_SIZE), actions_t]
                    loss_policy_v = -log_prob_actions_v.mean()

                    # add an entropy bonus to the loss function, it is negative so will reduce loss
                    prob_v = F.softmax(logits_v, dim=1)
                    entropy_loss_v = ENTROPY_BETA * (prob_v * log_prob_v).sum(dim=1).mean()

                    loss_v = entropy_loss_v + loss_value_v + loss_policy_v
                    loss_v.backward()
                    nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
                    optimizer.step()

                    tb_tracker.track("advantage", adv_v, step_idx)
                    tb_tracker.track("values", value_v, step_idx)
                    tb_tracker.track("batch_rewards", q_vals_v, step_idx)
                    tb_tracker.track("loss_entropy", entropy_loss_v, step_idx)
                    tb_tracker.track("loss_policy", loss_policy_v, step_idx)
                    tb_tracker.track("loss_value", loss_value_v, step_idx)
                    tb_tracker.track("loss_total", loss_v, step_idx)
    finally:
        for p in data_proc_list:
            p.terminate()
            p.join()
