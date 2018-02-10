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
from sklearn.neighbors import KNeighborsClassifier
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


if __name__ == "__main__":
  # Forward declaration of program flow
  N     = 5000
  a     = 0.25
  alpha = 3.14159/4
  nbr   = 5
  res   = []

  # Generate checkerboard
  X, y = gen_cb(N, a, alpha)

  # Create checkerboard plot
  plt.figure(figsize=(20,10))
  plt.subplot(2, 1, 1)
  plt.title("Generated Checkerboard")
  plt.plot(X[np.where(y==1)[0], 0], X[np.where(y==1)[0], 1], 'o')
  plt.plot(X[np.where(y==2)[0], 0], X[np.where(y==2)[0], 1], 's', c = 'r')
  plt.xlim(X[:,0].min(), X[:,0].max())
  plt.ylim(X[:,1].min(), X[:,1].max())

  # Create color mesh plot
  plt.subplot(2, 1, 2)
  plt.title("Color Plot of P(X|Y) using kNN")
  cm_bright = ListedColormap(['blue', 'red'])

  # Modifying arangement step to 0.01 (default is 1)
  x_arr = np.arange(X[:,0].min(), X[:,0].max(), 0.01)
  y_arr = np.arange(X[:,1].min(), X[:,1].max(), 0.01)
  x_grid, y_grid = np.meshgrid(x_arr, y_arr)

  # Apply density estimator KNN, fit the checkerboard data to the output results
  knn = KNeighborsClassifier(n_neighbors=nbr)
  knn.fit(X, y)

  # Prediction of X giving the fit to y above
  pred = knn.predict(np.c_[x_grid.ravel(), y_grid.ravel()]).reshape(x_grid.shape)
  plt.pcolormesh(x_grid, y_grid, pred, cmap=cm_bright, alpha=0.4)
  plt.plot(X[np.where(y==1)[0], 0], X[np.where(y==1)[0], 1], 'o')
  plt.plot(X[np.where(y==2)[0], 0], X[np.where(y==2)[0], 1], 's', c = 'r')
  plt.xlim(x_grid.min(), x_grid.max())
  plt.ylim(y_grid.min(), y_grid.max())

  # Display plots
  plt.show()
  
