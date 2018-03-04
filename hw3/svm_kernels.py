"""
Author: David Akre
Date: 3/3/18
Title: Support Vector Machines
Description: Generate a 2D Gaussian data set with at least two components 
(one for a positive class and one for a negative one). Train and test the 
classifier of two disjoint data sets generated from gaussian. Create two 
different kernels (one of which should be RBF and show the impact of the 
free parameter has on the decision boundary).
- Plot the training data, testing data with predicted class labels, and
indicate the classifier error
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Global declarations for this python script
cm_bright = ListedColormap(['red', 'blue'])

def gen_data():
  """
  gen_data : function that generates a 2D Guassian data set with positive and 
             negative components
  """
  X, y = make_blobs(n_samples=5000, centers=2, n_features=2, random_state=42)

  Xtr, Xtst, ytr, ytst = train_test_split(X, y, random_state=42)

  return X, y, Xtr, Xtst, ytr, ytst


def setup_plot_one(X, y, Xtr, Xtst, ytr, ytst):
  """
  setup_plot : function that sets up plot to be used for the classifiers
  """
  x_arr = np.arange(X[:,0].min(), X[:,0].max(), 0.1)
  y_arr = np.arange(X[:,1].min(), X[:,1].max(), 0.1)
  x_grid, y_grid = np.meshgrid(x_arr, y_arr)

  plt.figure(figsize=(20,10))
  ax = plt.subplot(2, 2, 1)
  ax.set_title("SVM - Training Data")

  # Plot training data points
  ax.scatter(Xtr[:,0], Xtr[:,1], c=ytr, cmap=cm_bright, alpha=0.8)
  return x_grid, y_grid

def setup_plot_ext(X, y, Xtr, Xtst, ytr, ytst, x_grid, y_grid, pred, ax, title):
  """
  setup_plot_two : function that sets up plot for the kernel rbf w/o free param (default)
  """
  ax.set_title(title)

  # Plot training and testing data
  ax.scatter(Xtr[:,0], Xtr[:,1], c=ytr, cmap=cm_bright, alpha=0.8)
  ax.scatter(Xtst[:,0], Xtst[:,1], c=ytst, cmap=cm_bright, alpha=0.8)
  ax.pcolormesh(x_grid, y_grid, pred, cmap=cm_bright, alpha=0.4)
  plt.xlim(x_grid.min(), x_grid.max())
  plt.ylim(y_grid.min(), y_grid.max())
  
 
if __name__ == "__main__":
  # Step 1 - Generate 2D Gaussian and split into test and training data
  X, y, Xtr, Xtst, ytr, ytst = gen_data()

  # Step 2 - Setup plots
  x_grid, y_grid = setup_plot_one(X, y, Xtr, Xtst, ytr, ytst)

  # Step 3 - Make predictions with SVM kernels and capture error

  # Step 3(a) - Make SVM RBF kernel prediction w/o free var
  clf = SVC(kernel='rbf') 
  clf.fit(X, y)
  pred = clf.predict(np.c_[x_grid.ravel(), y_grid.ravel()]).reshape(x_grid.shape)

  # Step 3(a) CONT'D - Plot results 
  ax = plt.subplot(2, 2, 2)
  setup_plot_ext(X, y, Xtr, Xtst, ytr, ytst, x_grid, y_grid, pred, ax, "SVM - RBF Kernel Prediction Default")

  # Step 3(a) CONT'd - Get accuracy of prediction
  score = clf.fit(X, y).score(Xtr, ytr)
  print "Error of the RBF Kernel w/ default settings = %0.2f" % (1 - score)

  # Step 3(b) - Make SVM RBF Kernel prediction showing free var difference
  # Updating free parameters C and epsilong to non-default values
  clf = SVC(C=0.01, kernel='rbf') 
  clf.fit(X, y)
  pred = clf.predict(np.c_[x_grid.ravel(), y_grid.ravel()]).reshape(x_grid.shape)

  # Step 3(b) CONT'D - Plot results 
  ax = plt.subplot(2, 2, 3)
  setup_plot_ext(X, y, Xtr, Xtst, ytr, ytst, x_grid, y_grid, pred, ax, "SVM - RBF Kernel Prediction C = 0.01")

  score = clf.fit(X, y).score(Xtr, ytr)
  print "Error of the RBF Kernel w/ C = 0.01 = %0.2f" % (1 - score)

  # Step 3(c) - Make SVM Linear Kernel prediction
  clf = SVC(kernel='linear') 
  clf.fit(X, y)
  pred = clf.predict(np.c_[x_grid.ravel(), y_grid.ravel()]).reshape(x_grid.shape)

  # Step 3(c) CONT'D - Plot results 
  ax = plt.subplot(2, 2, 4)
  setup_plot_ext(X, y, Xtr, Xtst, ytr, ytst, x_grid, y_grid, pred, ax, "SVM - Linear Kernel Prediction")

  score = clf.fit(X, y).score(Xtr, ytr)
  print "Error of the Linear SVM Kernel = %0.2f" % (1 - score)

  # Step 4 - Display results
  plt.show()

