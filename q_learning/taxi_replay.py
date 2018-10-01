# From The School of AI's Move 37 Course https://www.theschool.ai/courses/move-37-course/
# Coding demo by Colin Skow
from __future__ import print_function, division
from builtins import range
import sys
import numpy as np
import gym

GAME = 'Taxi-v2'
env = gym.make(GAME)
MAX_STEPS = env.spec.timestep_limit

if __name__ == '__main__':
  if len(sys.argv) < 2:
    sys.exit('Must specify a checkpoint file in command line')
  cp_file = sys.argv[1]
  
  Q = np.load(cp_file)
  total_reward = 0
  state = env.reset()
  env.render()

  for step in range(MAX_STEPS):
    prevState = state
    action = np.argmax(Q[state])
    state, reward, done, info = env.step(action)
    total_reward += reward
    env.render()
    if done :
        break

  print('Total reward:', total_reward)
