import gym
import ptan
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from tensorboardX import SummaryWriter

from lib import common
from lib.doom_wrappers import make_doom_env
from lib.model import NoisyDuelingDQN
from configs import PARAMS


def calc_loss(batch, batch_weights, net, tgt_net, gamma, device="cpu", double=True):
    states, actions, rewards, dones, next_states = common.unpack_batch(batch)

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device)
    batch_weights_v = torch.tensor(batch_weights).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    if double:
        next_state_actions = net(next_states_v).max(1)[1]
        next_state_values = tgt_net(next_states_v).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
    else:
        next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0

    expected_state_action_values = next_state_values.detach() * gamma + rewards_v
    losses_v = batch_weights_v * (state_action_values - expected_state_action_values) ** 2
    return losses_v.mean(), losses_v + 1e-5


if __name__ == "__main__":
    common.mkdir('.', 'checkpoints')
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("--config", default='pong', help="Config to use")
    args = parser.parse_args()
    params = PARAMS[args.config]
    device = torch.device("cuda" if args.cuda else "cpu")

    if args.config == 'doom':
        import vizdoomgym
        env = make_doom_env(params['env_name'])
    else:
        env = gym.make(params['env_name'])
        env = ptan.common.wrappers.wrap_dqn(env)

    writer = SummaryWriter(comment="-" + params['run_name'] + "-rainbow")
    net = NoisyDuelingDQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = ptan.agent.TargetNet(net)
    agent = ptan.agent.DQNAgent(net, ptan.actions.ArgmaxActionSelector(), device=device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params['gamma'], steps_count=params['n_steps'])
    buffer = common.PrioReplayBuffer(exp_source, params['replay_size'], params['prio_replay_alpha'])
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])

    frame_idx = 0

    with common.RewardTracker(writer, params['stop_reward']) as reward_tracker:
        while True:
            frame_idx += 1
            buffer.populate(1)
            beta = min(1.0, params['beta_start'] + frame_idx * (1.0 - params['beta_start']) / params['beta_frames'])

            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
              writer.add_scalar("beta", beta, frame_idx)
              finished, save_checkpoint = reward_tracker.reward(new_rewards[0], frame_idx)
              if save_checkpoint:
                  torch.save(net.state_dict(), './checkpoints/' + params['env_name'] + "-best.dat")
              if finished:
                  break

            if len(buffer) < params['replay_initial']:
                continue

            optimizer.zero_grad()
            batch, batch_indices, batch_weights = buffer.sample(params['batch_size'], beta)
            loss_v, sample_prios_v = calc_loss(batch, batch_weights, net, tgt_net.target_model,
                                               params['gamma'], device=device)
            loss_v.backward()
            optimizer.step()
            buffer.update_priorities(batch_indices, sample_prios_v.data.cpu().numpy())

            if frame_idx % params['target_net_sync'] == 0:
                tgt_net.sync()

            if frame_idx % 500 == 0:
                for layer_idx, sigma_l2 in enumerate(net.noisy_layers_sigma_snr()):
                    writer.add_scalar("sigma_snr_layer_%d" % (layer_idx+1),
                                      sigma_l2, frame_idx)
