"""
Author: David AKre
Date: 4/14/18
Title: Self-Training Gaussian
Description: Generate a 2D Gaussian data set and then
implement a self-training algorithm that reports 
"""

import numpy as np
import pandas as pd
from sklearn.semi_supervised import LabelPropagation
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

def self_train(Xtr, labels, random_unlabeled_points):
  # Forward declaration
  err    = 0
  count  = 0
  labels = np.split(labels, 5)

  # Split training data into 5 chunks to iterate through for self-training
  for x in np.split(Xtr, 5):
    clf = GaussianNB().fit(x, labels[count])
    err += 1 - clf.score(x, labels[count])

    if count == 0:
      print "Error at the first pass through - %0.2f" % err

    clf = KNeighborsClassifier(n_neighbors = 5).fit(x, labels[count])
    err += 1 - clf.score(x, labels[count])

    if count == 3:
      print "Error half way through - %0.2f" % (err / 6)

    count += 1

  return err


def gen_data():
  """
  gen_data : function that generates a 2D Guassian data set with positive and 
             negative components
  """
  X, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=42)
  Xtr, Xtst, ytr, ytst = train_test_split(X, y, random_state=42)

  return X, y, Xtr, Xtst, ytr, ytst


if __name__ == "__main__":
  # Part 1 - 10% labeled
  # Generate 2D Gaussian and split into test and training data
  X, y, Xtr, Xtst, ytr, ytst = gen_data()

  print "Part 1 - Error of self training algorithm with 10% of the data labeled"

  # Generate some randomness to our label/unlabel split from created data (%10)
  rng = np.random.RandomState(42)
  random_unlabeled_points = rng.rand(len(ytr)) < 0.9
  label_prop_model = LabelPropagation()
  labels = np.copy(ytr)
  labels[random_unlabeled_points] = -1
  label_prop_model.fit(Xtr, labels)

  err = self_train(Xtr, labels, random_unlabeled_points)
  print "Error at the end - %0.2f" % (err / 10)

  # Part 2 - 25% labeled
  # Generate 2D Gaussian and split into test and training data
  X, y, Xtr, Xtst, ytr, ytst = gen_data()

  print "Part 2 - Error of self training algorithm with 25% of the data labeled"

  # Generate some randomness to our label/unlabel split from created data (25%)
  random_unlabeled_points = rng.rand(len(ytr)) < 0.75
  labels = np.copy(ytr)
  labels[random_unlabeled_points] = -1
  label_prop_model.fit(Xtr, labels)

  err = self_train(Xtr, labels, random_unlabeled_points)
  print "Error at the end - %0.2f" % (err / 10)

