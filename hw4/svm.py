"""
Author: David Akre
Date: 3/29/18
Title: SVM Dual Domain - CVX
Description: This python script will implement the 
dual form SVM dervied in the theory section prob 1
utilizing CVX
"""

import numpy as np
import pandas as pd
from cvxpy import *
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

def main():
  # Forward declarations for program flow
  source_test  = "svm_data/source_test.csv"
  source_train = "svm_data/source_train.csv"
  target_test  = "svm_data/target_test.csv"
  target_train = "svm_data/target_train.csv"
  cm_bright    = ListedColormap(['red', 'blue'])

  # Step 1 - Read data
  # Read src tr
  dataset = pd.read_csv(source_train)
  X_src_tr = dataset.values[:,:-1]
  y_src_tr = dataset.values[:,-1]
 
  # Read src tst
  dataset = pd.read_csv(source_test)
  X_src_tst = dataset.values[:,:-1]
  y_src_tst = dataset.values[:,-1]

  # Read tgt tr
  dataset = pd.read_csv(target_train)
  X_tgt_tr = dataset.values[:,:-1]
  y_tgt_tr = dataset.values[:,-1]

  # Read tft tst
  dataset = pd.read_csv(target_test)
  X_tgt_tst = dataset.values[:,:-1]
  y_tgt_tst = dataset.values[:,-1]

  # Setup cvx variables for dual form SVM
  C = Parameter(sign="positive") 
  B = Parameter(sign="positive")
  zeta = Variable()
  count = 0

  # Training data
  constraints = [
    zeta >= 0,
    zeta >= 0  
  ]

  n = []
  for tgt in X_tgt_tr.ravel():
    n.append(0.5 * norm(tgt, 2))

  o = []
  for tgt in X_tgt_tr.ravel().T:
    for src in X_src_tr.ravel():
      #o.append(B * tgt * src)
      o.append(tgt * src)

  c = [] 
  s = []
  for y in y_tgt_tr.ravel():
    tmp = 0
    for t in X_tgt_tr.T.ravel():
      tmp = y * t

    for x in X_tgt_tr.ravel():
      tmp *= x

    #constraints[0] = (tmp + y * B) >= (1 - zeta)
    constraints[0] = (tmp + y) >= (1 - zeta)
    s.append(Problem(Minimize(n[count] + o[count]), constraints).solve())
    count += 1

  x_tgt_arr = np.arange(X_tgt_tr.min(), X_tgt_tr.max())
  x_src_arr = np.arange(X_src_tr.min(), X_src_tr.max())
  y_tgt_arr = np.arange(X_src_tr.min(), X_src_tr.max())
  y_src_arr = np.arange(X_src_tr.min(), X_src_tr.max())

  x_tgt_grid, y_tgt_grid = np.meshgrid(x_tgt_arr, y_tgt_arr)
  x_src_grid, y_src_grid = np.meshgrid(x_src_arr, y_src_arr)

  ax = plt.subplot(2, 3, 1)
  ax.set_title("SVM Target (Training Data)")
  ax.set_xlim(x_tgt_grid.min(), x_tgt_grid.max())
  ax.set_ylim(y_tgt_grid.min(), y_tgt_grid.max())
  ax.scatter(X_tgt_tr[:,0], X_tgt_tr[:,1], c=y_tgt_tr, cmap=cm_bright, alpha=0.8)

  ax = plt.subplot(2, 3, 2)
  ax.set_title("SVM Source (Traning Data)")
  ax.set_xlim(x_src_grid.min(), x_src_grid.max())
  ax.set_ylim(y_src_grid.min(), y_src_grid.max())
  ax.scatter(X_src_tr[:,0], X_src_tr[:,1], c=y_src_tr, cmap=cm_bright, alpha=0.8)

  ax = plt.subplot(2, 3, 3)
  ax.set_title("SVM Dual Form Results (Traning Data)")
  ax.set_xlim(x_src_grid.min(), x_src_grid.max())
  ax.set_ylim(y_src_grid.min(), y_src_grid.max())
  ax.scatter(X_tgt_tr[:,0], s, linestyle='--', cmap=cm_bright, alpha=0.8)

  # Testing Data
  n = []
  for tgt in X_tgt_tst.ravel():
    n.append(0.5 * norm(tgt, 2))

  o = []
  for tgt in X_tgt_tst.ravel().T:
    for src in X_src_tst.ravel():
      #o.append(B * tgt * src)
      o.append(tgt * src)

  c = [] 
  s = []
  for y in y_tgt_tst.ravel():
    tmp = 0
    for t in X_tgt_tst.T.ravel():
      tmp = y * t

    for x in X_tgt_tst.ravel():
      tmp *= x

    #constraints[0] = (tmp + y * B) >= (1 - zeta)
    constraints[0] = (tmp + y) >= (1 - zeta)
    s.append(Problem(Minimize(n[count] + o[count]), constraints).solve())
    count += 1

  x_tgt_arr = np.arange(X_tgt_tst.min(), X_tgt_tst.max())
  x_src_arr = np.arange(X_src_tst.min(), X_src_tst.max())
  y_tgt_arr = np.arange(X_src_tst.min(), X_src_tst.max())
  y_src_arr = np.arange(X_src_tst.min(), X_src_tst.max())

  x_tgt_grid, y_tgt_grid = np.meshgrid(x_tgt_arr, y_tgt_arr)
  x_src_grid, y_src_grid = np.meshgrid(x_src_arr, y_src_arr)

  ax = plt.subplot(2, 3, 4)
  ax.set_title("SVM Target (Testing Data)")
  ax.set_xlim(x_tgt_grid.min(), x_tgt_grid.max())
  ax.set_ylim(y_tgt_grid.min(), y_tgt_grid.max())
  ax.scatter(X_tgt_tst[:,0], X_tgt_tst[:,1], c=y_tgt_tst, cmap=cm_bright, alpha=0.8)

  ax = plt.subplot(2, 3, 5)
  ax.set_title("SVM Source (Testing Data)")
  ax.set_xlim(x_src_grid.min(), x_src_grid.max())
  ax.set_ylim(y_src_grid.min(), y_src_grid.max())
  ax.scatter(X_src_tst[:,0], X_src_tst[:,1], c=y_src_tst, cmap=cm_bright, alpha=0.8)

  ax = plt.subplot(2, 3, 6)
  ax.set_title("SVM Dual Form Results (Testing Data)")
  ax.set_xlim(x_src_grid.min(), x_src_grid.max())
  ax.set_ylim(y_src_grid.min(), y_src_grid.max())
  ax.scatter(X_tgt_tst[:,0], s, linestyle='--', cmap=cm_bright, alpha=0.8)

  plt.show()


if __name__ == "__main__":
  main()  

