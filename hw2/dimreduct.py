"""
Author: David Akre
Date: 2/7/18
Title: Dimensionality Reduction
Description: This python script will read in 10 different
data sets from the ECE523 repository and perform a naive
bayes classifier on the data with and without a preprocessor
algorithm which will be PCA (i.e. Principal Component 
Analysis) in this circumstance
"""

import numpy as np
import pandas as pd
from random import *
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA

# Step 1 - Read in data
def read_data(file_name):
  # Forward declarations
  dir_name = "data/"
  dataset = pd.read_csv(dir_name + file_name)
  X = dataset.values[:,:-1]
  y = dataset.values[:,-1]
  size = dataset.columns.size

  return X, y, size


# Step 2(a) - Apply PCA preprocessor on data
def apply_pca(X):
  # Creating PCA object and setting n_comp to 0.5 to invoke mle choice of dimensionality
  pca = PCA(n_components = 0.5)
  return pca.fit_transform(X)


# Step 2(b) - Apply Naive Bayes classification on data
def apply_nb(X, y, size):
  gnb = GaussianNB()
  y_pred = gnb.fit(X, y).predict(X)

  return y_pred


if __name__ == "__main__":
  # Forward declaration for program flow
  results = {}
  file_list = [
    "car.csv",
    "lung-cancer.csv",
    "lymphography.csv",
    "magic.csv",
    "waveform.csv",
    "wine.csv",
    "wine-quality-red.csv",
    "wine-quality-white.csv",
    "yeast.csv",
    "zoo.csv"
  ]

  for i in range(len(file_list)):
    # Initially read in data for the program to flow properly 
    X, y, size = read_data(file_list[i])

    # Part 1 - Apply naive bayes classifier on data w/o PCA
    y_pred = apply_nb(X, y, size)
    accuracy = round(1 - (np.float((y != y_pred).sum()) / np.float(y.shape[0])), 5)
    results[file_list[i]] = [accuracy]
   
    # Part 2 - Apply PCA preprocessing and then naive bayes classification
    X = apply_pca(X)
    y_pred = apply_nb(X, y, size)
    accuracy = round(1 - (np.float((y != y_pred).sum()) / np.float(y.shape[0])), 5)
    results[file_list[i]].append(accuracy)
 
  # Part 3 - Write results to csv file
  df = pd.DataFrame.from_dict(results, orient="index")
  df.columns = ["Naive Bayes Classifier", "Naive Bayes Classifier with PCA preprocessing"]
  df.to_csv("results/results.csv")
  
