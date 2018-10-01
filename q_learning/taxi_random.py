# From The School of AI's Move 37 Course https://www.theschool.ai/courses/move-37-course/
# Coding demo by Colin Skow
from __future__ import print_function, division
from builtins import range
import time, random, math
import numpy as np
import gym

GAME = 'Taxi-v2'
env = gym.make(GAME)

MAX_STEPS = env.spec.timestep_limit
total_reward = 0

state = env.reset()
env.render()

for step in range(MAX_STEPS):
  action = env.action_space.sample()
  state, reward, done, info = env.step(action)
  total_reward += reward
  env.render()
  if done :
      break

print('Total reward:', total_reward)
