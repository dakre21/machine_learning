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
  probs = []
  for y in y_tgt_tr.ravel():
    tmp = 0
    for t in X_tgt_tr.T.ravel():
      tmp = y * t

    for x in X_tgt_tr.ravel():
      tmp *= x

    #constraints[0] = (tmp + y * B) >= (1 - zeta)
    constraints[0] = (tmp + y) >= (1 - zeta)
    probs.append(Problem(Minimize(n[count] + o[count]), constraints))
    count += 1

  for p in probs:
    print p.solve()



  """
  eq1 = 0.5 * norm(X_tgt_tr, 2)
  eq2 = C * sum_entries(zeta)
  eq3 = B * np.outer(X_tgt_tr.T.ravel(), X_src_tr.ravel())
  exp_tr = eq1 + eq2 - eq3
  prob_tr = Problem(Minimize(exp_tr), X_tgt_tr.ravel())
  """

  """
  #alpha = Variable()
  alpha = Parameter(sign="positive")

  mul_one = np.outer(X_src_tr.ravel(), y_src_tr.ravel())
  mul_two = np.outer(y_tgt_tr.ravel(), X_tgt_tr.T.ravel())
  prod = np.outer(mul_one, mul_two)
  df_tr_one = alpha * (1 - mul_elemwise(B, prod))
  #df_tr_one = alpha * (1 - B * prod)

  mul_one = np.outer(y_tgt_tr.ravel(), X_tgt_tr.ravel())
  mul_two = np.outer(y_tgt_tr.ravel(), X_tgt_tr.T.ravel())
  prod = np.outer(mul_one, mul_two)
  df_tr_two = 0.5 * sum_entries(sum_entries(mul_elemwise(alpha, alpha * prod)))
  #df_tr_two = 0.5 * sum_entries(sum_entries(alpha * alpha * prod))

  mul_one = np.outer(X_src_tst.ravel(), y_src_tst.ravel())
  mul_two = np.outer(y_tgt_tst.ravel(), X_tgt_tst.T.ravel())
  prod = np.dot(mul_one, mul_two)
  df_tst_one = alpha * (1 - mul_elemwise(B, prod))
  #df_tst_one = alpha * (1 - B * prod)

  mul_one = np.outer(y_tgt_tst.ravel(), X_tgt_tst.ravel())
  mul_two = np.outer(y_tgt_tst.ravel(), X_tgt_tst.T.ravel())
  prod = mul_one * mul_two
  df_tst_one = alpha * (1 - mul_elemwise(B, prod))
  df_tr_two = 0.5 * sum_entries(sum_entries(mul_elemwise(alpha, alpha * prod)))
  #df_tst_two = 0.5 * sum_entries(sum_entries(alpa * alpha * prod))
  """

if __name__ == "__main__":
  main()  

