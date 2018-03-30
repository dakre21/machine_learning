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
  alpha = Variable()
  zeta = Variable()

  mul_one = np.outer(X_src_tr.ravel(), y_src_tr.ravel())
  mul_two = np.outer(y_tgt_tr.ravel(), X_tgt_tr.T.ravel())
  prod = np.outer(mul_one, mul_two)
  df_tr_one = alpha * (1 - B * prod)

  mul_one = np.outer(y_tgt_tr.ravel(), X_tgt_tr.ravel())
  mul_two = np.outer(y_tgt_tr.ravel(), X_tgt_tr.T.ravel())
  prod = np.outer(mul_one, mul_two)
  df_tr_two = 0.5 * sum_entries(sum_entries(alpha * alpha * prod))

  mul_one = np.outer(X_src_tr.ravel(), y_src_tr.ravel())
  mul_two = np.outer(y_tgt_tr.ravel(), X_tgt_tr.T.ravel())
  prod = np.outer(mul_one, mul_two)
  df_tst_one = alpha * (1 - B * prod)

  mul_one = np.outer(y_tgt_tst.ravel(), X_tgt_tst.ravel())
  mul_two = np.outer(y_tgt_tst.ravel(), X_tgt_tst.T.ravel())
  prod = np.outer(mul_one, mul_two)
  df_tst_two = 0.5 * sum_entries(sum_entries(alpa * alpha * prod))

  # Minimize Training Data
  prob_tr = Problem(Minimize(df_tr_one - df_tr_two))

  # Minimize Testing Data
  prob_tst = Problem(Minimize(df_tst_one - df_tst_two))

  # Solve each problem
  print prob_tr.solve()
  print prob_tst.solve()


if __name__ == "__main__":
  main()  

