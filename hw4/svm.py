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


if __name__ == "__main__":
  main()  

