"""
Author: David Akre
Date: 3/29/18
Title: Ensembles
Description: This python script will evaluate AdaBoosting
vs Bagging w.r.t its testing error on 15 different data
sets
"""

from os import listdir
import numpy as np
import pandas as pd
from random import *
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier

# Global declarations
dataset = []
X       = []
y       = []

def read_data(file_path="data/"):
  # Forward decls
  count = 0
  files = listdir(file_path)

  for f in files:
    dataset.append(pd.read_csv(file_path + f)) 
    X.append(dataset[count].values[:,:-1])
    y.append(dataset[count].values[:,-1])
    count += 1

  return files


if __name__ == "__main__":
  # Forward declarations for program flow
  cm_bright = ListedColormap(['red', 'blue'])
  x_axis    = []
  file_path = "data"
  size      = 2 

  # Step 1 - Read in the data
  files = read_data()

  # Setup plot
  plt.figure(figsize=(20, 10))
  plt.suptitle("Testing Error Sweeping N from 2 to 9 - AdaBoost vs Bagging")

  print "X-Axis Identifier"
  for i in range(len(files)):
    print "x = " + str(i) + " coordinates with " + files[i]
    x_axis.append(i)

  # One to one ratio (X to Y)
  plt.xlim(0, len(x_axis))
  plt.ylim(0, len(x_axis))

  # Step 2 - Sweep from 2 to 10 ensemble sizes
  for i in range(8):
    ada_err = []
    bag_err = []
    # Step 3 - Gather testing error
    for j in range(len(files)):
      # Step 3(a) - Apply AdaBoosting 
      clf = AdaBoostClassifier(n_estimators = size).fit(X[j], y[j])
      ada_err.append(1-clf.score(X[j], y[j]))

      # Step 3(b) - Apply Bagging
      clf = BaggingClassifier(n_estimators = size).fit(X[j], y[j])
      bag_err.append(1-clf.score(X[j], y[j]))


    # Step 4 - Plot testing error from 15 data sets
    ax = plt.subplot(2, 4, 1+i)
    ax.set_title("Adaboost vs Bagging for N = " + str(size) + " Estimators")
    ax.scatter(x_axis, ada_err, cmap=cm_bright, alpha=0.8)
    ax.scatter(x_axis, bag_err, cmap=cm_bright, alpha=0.8)
    size += 1

  # Step 5 - Display Plot
  plt.show()
