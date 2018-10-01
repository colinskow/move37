# From The School of AI's Move 37 Course https://www.theschool.ai/courses/move-37-course/
# Coding demo by Colin Skow
import os
import numpy as np

def mkdir(name):
  base = os.getcwd()
  path = os.path.join(base, name)
  if not os.path.exists(path):
    os.makedirs(path)
  return path

# https://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy
def moving_average(values, n=100) :
  ret = np.cumsum(values, dtype=float)
  ret[n:] = ret[n:] - ret[:-n]
  return ret[n - 1:] / n

def checkpoint(data, dir, filename, step):
  path = mkdir(dir)
  file_path = os.path.join(path, filename + '_' + str(step) + '.npy')
  np.save(file_path, data)
  return file_path
