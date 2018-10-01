# From The School of AI's Move 37 Course https://www.theschool.ai/courses/move-37-course/
# Coding demo by Colin Skow
from __future__ import print_function, division
from builtins import range
import random
import numpy as np
import matplotlib.pyplot as plt
import gym
from utils import checkpoint, mkdir, moving_average

GAME = 'Taxi-v2'
env = gym.make(GAME)
CHECKPOINT_DIR = 'checkpoints'
MAX_STEPS = env.spec.timestep_limit # maximum steps in an episode, 200 for Taxi-v2

NUM_EPISODES = 50000
GAMMA = 0.95  # Discount factor from Bellman Equation
START_ALPHA = 0.1  # Learning rate, how much we update our Q table each step
ALPHA_TAPER = 0.01 # How much our adaptive learning rate is adjusted each update
START_EPSILON = 1.0  # Probability of random action
EPSILON_TAPER = 0.01 # How much epsilon is adjusted each step

obs_dim = env.observation_space.n # size of our state
action_dim = env.action_space.n # number of actions
# Initialize our Q table
Q = np.zeros((obs_dim, action_dim))

# we're going to keep track of how many times each Q(s,a) is updated
# this is for our adaptive learning rate
state_visit_counts = {}
Q = np.zeros((obs_dim, action_dim))
update_counts = np.zeros((obs_dim, action_dim), dtype=np.dtype(int))

def update_Q(prev_state, action, reward, cur_state):
  
  alpha = START_ALPHA / (1.0 + update_counts[prev_state][action] * ALPHA_TAPER)
  update_counts[prev_state][action] += 1
  Q[prev_state][action] += \
    alpha * (reward + GAMMA * max(Q[cur_state]) - Q[prev_state][action])


def epsilon_action(s, eps=START_EPSILON):
  if random.random() < (1 - eps):
    return np.argmax(Q[s])
  else:
    return env.action_space.sample()
  

if __name__ == '__main__':
  print("\nObservation\n--------------------------------")
  print("Shape :", obs_dim)
  print("\nAction\n--------------------------------")
  print("Shape :", action_dim, "\n")

  total_reward = 0
  deltas = []

  for episode in range(NUM_EPISODES + 1):

    # Taper EPSILON each episode
    eps = START_EPSILON / (1.0 + episode * EPSILON_TAPER)

    if episode % 1000 == 0:
      print("Episode =", episode, " |  Avg Reward =", total_reward/1000, " | Epsilon =", eps)
      total_reward = 0

    if episode % 10000 == 0:
      cp_file = checkpoint(Q, CHECKPOINT_DIR, GAME, episode)
      print("Saved checkpoint to: ", cp_file)

    biggest_change = 0
    curr_state = env.reset()
    for step in range(MAX_STEPS):
      prev_state = curr_state
      state_visit_counts[prev_state] = state_visit_counts.get(prev_state,0) + 1
      action = epsilon_action(curr_state, eps)
      curr_state, reward, done, info = env.step(action)
      total_reward += reward
      old_qsa = Q[prev_state][action]
      update_Q(prev_state, action, reward, curr_state)
      biggest_change = max(biggest_change, np.abs(old_qsa - Q[prev_state][action]))
      if done:
        break

    deltas.append(biggest_change)
  
  mean_state_visits = np.mean(list(state_visit_counts.values()))
  print("each state was visited on average: ", mean_state_visits, " times.")

  plt.plot(moving_average(deltas, n=1000))
  plt.show()
