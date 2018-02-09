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
from math import *
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
def sgd(W, X, y, size):
  # Forward declarations for sgd algo
  wnew = 0.5 # Set initial weight to 0.5
  T    = 10  # Set max iterations to 10
  wold = 0   # Initialize wold to 0
  n    = 1   # Set learning rate to 1
  eps  = 0.5 # User defined epsilon to check for convergence

  for t in range(T):
    # Shuffle data
    for z in range(size):
      W[z] = W[randint(1,size-1)]

    # Utilizing online batch learning for sgd
    for i in range(size):
      wold = wnew
      wnew = wold - n * (-2 * (y[i] - X[i]) * X[i]) 
      # Check for convergence
      if math.sqrt(math.pow(wnew - wold, 2)) < eps:
        break
    

# Step 3 - Plot results
def plot(W, X, y, size):
  # Get dimensions of matrix
  x_arr = np.arrange(X[:,0].min(), X[:,0].max()
  y_arr = np.arrange(X[:,1].min(), X[:,1].max()
  x_grid, y_grid = np.meshgrid(x_arr, y_arr)
  cm_bright = ListedColormap(['red', 'blue'])

  # Setup plot
  ax = plt.subplot()
  ax.set_title("SGD Logistic Regression from miniboone data")
  ax.set_xlim(x_grid.min(), x_grid.max())
  ax.set_ylim(y_grid.min(), y_grid_max())
  ax.scatter(X[:,0], X[:,1], c=y, cmap=cm_bright, alpha=0.8)
  ax.contourf(x_grid, y_grid, 


if __name__ == "__main__":
  # Fordward declarations for program flow
  filepath = "data/miniboone.csv"
  W        = sample(range(0,1), size) # Generate training data

  X, y, size = read_data(filepath)
  sgd(W, X, y, size)
  plot(W, X, y, size)
