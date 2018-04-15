"""
Author: David Akre
Date: 3/29/18
Title: Self-Training KFold Crossing
Description: This python script will perform the following:
1) Perform self-training on 15% of the data (labeled/unlabeled) 
which is randomly selected (using NB as the supervised training algo
and density estimation for the unsupervised training algo)
2) Report results using 5-Fold Crossing
"""

from os import listdir
from random import randint
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


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
  file_path = "data"
  sup_len_x = 0
  index     = 0

  # Step 1 - Read in the data
  files = read_data()

  # Step 2 - For each data set perform self-training
  for i in range(len(files)):
    # Initialize scores
    sup_len = int(0.15 * len(X[i]))
    score_sup = 0
    score_un  = 0
    tmp_un_x = np.zeros(0)
    tmp_un_y = np.zeros(0)
    tmp_sup_x = np.zeros(0)
    tmp_sup_y = np.zeros(0)

    # Log file
    print "**********************************************"
    print "File: " + files[i]

    # Train 15% of labeled data (random)
    index = randint(0, len(X[i]) - sup_len)
    for j in range(len(X[i])):
      if index <= j <= index + sup_len:
        tmp_sup_x = np.append(tmp_sup_x, X[i][j])
      else:
        tmp_un_x = np.append(tmp_un_x, (X[i][j]))

    for j in range(len(y[i])):
      if index <= j <= index + sup_len:
        tmp_sup_y = np.append(tmp_sup_y, y[i][j])
      else:
        tmp_un_y = np.append(tmp_un_y, (y[i][j]))

    # Train supervised
    tmp_sup_x = np.reshape(tmp_sup_x, (len(tmp_sup_y), -1))
    kf = KFold(n_splits = 5).split(tmp_sup_x)
    for test, train in kf:
      clf = GaussianNB().fit(tmp_sup_x[train], tmp_sup_y[train])
      score_sup += cross_val_score(clf, tmp_sup_x, tmp_sup_y)[0]

    # Predict unsupervised
    tmp_un_x = np.reshape(tmp_un_x, (len(tmp_un_y), -1))
    kf = KFold(n_splits = 5).split(tmp_un_x)
    for test, train in kf:
      clf = KNeighborsClassifier(n_neighbors = 5).fit(tmp_un_x[train], tmp_un_y[train])
      score_un += cross_val_score(clf, tmp_un_x, tmp_un_y)[0]

    # Report accuracy for process
    score_sup = score_sup / 5
    score_un = score_un / 5

    # Resport results
    print "Supervised Learning Accuracy: %0.2f" % score_sup
    print "Unsupervised Learning Accuracy: %0.2f" % score_un
    print "Total Semi Supervised Learning Accuracy: %0.2f" % ((score_sup + score_un) / 2)


