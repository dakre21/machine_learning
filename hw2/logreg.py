"""
Author: David Akre
Date: 2/7/18
Title: Logistic Regression utilizing SGD
Description: This python script contains an algorithm which
utilizes SGD to perform logistic regression on a dataset
"""

import numpy as np
import pandas as pd
from random import *
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

# Step 1 - Read in the dataset
def read_data(filepath):
  dataset = pd.read_csv(filepath)
  X = dataset.values[:,:-1]
  y = dataset.values[:,-1]
  size = dataset.columns.size

  return X, y, size

# Step 2 - Apply SGD on data
def sgd(X, y, size):
  # Forward declarations for sgd algo
  wnew     = 0.5  # Set initial weight to 0.5
  wold     = 0    # Initialize wold to 0
  n        = 0.5  # Set learning rate to 1
  eps      = 0.5  # User defined epsilon to check for convergence

  for t in range(size):
    # Shuffle data
    for z in range(size):
      X[z] = X[randint(1,size-1)]

    # Utilizing online batch learning for sgd
    for i in range(size):
      wold = wnew
      wnew = wold - n * (-2 * (y[i] - wold * X[t][i]) * X[t][i]) 

      # Check for convergence
      if np.abs(wnew - wold) < eps:
        break

  return wnew
    

# Step 3 - Plot results
def setup_plot(X, y, size):
  # Get dimensions of matrix
  x_arr = np.arange(X.min(), X.max())
  y_arr = np.arange(X.min(), X.max())
  x_grid, y_grid = np.meshgrid(x_arr, y_arr)
  cm_bright = ListedColormap(['red', 'blue'])

  # Setup plot
  ax = plt.subplot()
  ax.set_title("SGD Logistic Regression from wine data")
  ax.set_xlim(x_grid.min(), x_grid.max())
  ax.set_ylim(y_grid.min(), y_grid.max())
  ax.scatter(X[:,0], X[:,1], c=y, cmap=cm_bright, alpha=0.8)

  return ax, np.int(x_grid.min())
 

if __name__ == "__main__":
  # Fordward declarations for program flow
  filepath = "data/wine.csv"

  X, y, size = read_data(filepath)
  ax, x_min = setup_plot(X, y, size)
  w = sgd(X, y, size)

  # Fit the classifer to the linear model
  x = []
  x_min = x_min - 1
  for i in range(size + 1):
    x.append(x_min + i)
    
  y = []
  for i in range(size + 1):
    y.append(1 / (1 + (np.exp(w) * i)))

  ax.plot(x, y, 'y--')
  plt.show()
