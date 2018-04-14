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
    score_sup = 0
    score_un  = 0

    # Perform KFold Crossing
    kf = KFold(n_splits = 5).split(X[i])
    for test, train in kf:
        sup_len = int(0.15 * len(X[i][train]))
        tmp_un_x = np.zeros(0)
        tmp_un_y = np.zeros(0)
        tmp_sup_x = np.zeros(0)
        tmp_sup_y = np.zeros(0)

        # Train 15% of labeled data (random)
        index = randint(0, len(X[i]) - sup_len)
        for j in range(len(X[i])):
          if index <= j <= index + sup_len:
            tmp_sup_x = np.append(tmp_sup_x, X[i][j])
            tmp_sup_y = np.append(tmp_sup_y, X[i][j])

          tmp_un_x = np.append(tmp_un_x, (X[i][j]))
          tmp_un_y = np.append(tmp_un_y, (y[i][j]))

        clf = GaussianNB().fit(X[i][train], y[i][train])
        score_sup += cross_val_score(clf, [tmp_sup_x], [tmp_sup_y])

        # Make predictions on the remaining unlabeled data
        clf = KNeighborsClassifier(n_neighbors = 5).fit(tmp_un_x, tmp_un_y)
        score_un += cross_val_score(clf, tmp_un_x, tmp_un_y)

    # Report accuracy for process
    score_sup = score_sup / 5
    score_un = score_un / 5

    print "*******************************************"
    print "File: " + files[i]
    print "Supervised Learning Accuracy: " + str(score_sup)
    print "Unsupervised Learning Accuracy: " + str(score_un)
    print "Total Semi Supervised Learning Accuracy: " + str(score_sup + score_un)


