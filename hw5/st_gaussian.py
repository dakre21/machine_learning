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
from sklearn.svm import SVC

def gen_data():
  """
  gen_data : function that generates a 2D Guassian data set with positive and 
             negative components
  """
  X, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=42)
  Xtr, Xtst, ytr, ytst = train_test_split(X, y, random_state=42)

  return X, y, Xtr, Xtst, ytr, ytst


if __name__ == "__main__":
  # Step 1 - Generate 2D Gaussian and split into test and training data
  X, y, Xtr, Xtst, ytr, ytst = gen_data()

  # Generate some randomness to our label/unlabel split from created data
  rng = np.random.RandomState(42)
  random_unlabeled_points = rng.rand(len(ytr)) < 0.3
  label_prop_model = LabelPropagation()
  labels = np.copy(ytr)
  labels[random_unlabeled_points] = -1
  label_prop_model.fit(Xtr, labels)

  # Split training data into 5 chunks to iterate through for self-training
  for x in np.split(Xtr, 5):
    print x
