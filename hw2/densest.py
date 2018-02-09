"""
Author: David Akre
Date: 2/7/18
Title: Density Estimation
Description: This python script will generate a checkerboard
of data from two classes and then use a density estimator
on the posterior P(Y|X) and then plot P(X|Y) on a color plot
"""

import numpy as np
import pandas as pd
from random import *
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA

# Step 1 - Generate checkerboard data
# NOTE: Utilizing ECE 523 gen_cb function to generate checkerboard
def gen_cb(N, a, alpha):
  d = np.random.rand(N, 2).T
  d_transformed = np.array([d[0]*np.cos(alpha)-d[1]*np.sin(alpha),
                            d[0]*np.sin(alpha)+d[1]*np.cos(alpha)]).T
  s = np.ceil(d_transformed[:,0]/a)+np.floor(d_transformed[:,1]/a)
  lab = 2 - (s%2)
  data = d.T

  return data, lab


# Step 2 - Apply a density estimator to classify the posterior
def dens_est():
  pass


if __name__ == "__main__":
  # Forward declaration of program flow
  N     = 5000
  a     = 0.25
  alpha = 3.14159/4

  X, y = gen_cb(N, a, alpha)

  plt.figure()
  plt.plot(X[np.where(y==1)[0], 0], X[np.where(y==1)[0], 1], 'o')
  plt.plot(X[np.where(y==2)[0], 0], X[np.where(y==2)[0], 1], 's', c = 'r')

  plt.show()
  
