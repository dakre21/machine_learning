""" 
Author: David Akre
Date: 4/15/18
Title: Market Predictor Model Controller
"""
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression


class ModelController:
  """
  Class ModelController :  Class defines the functionality carried out by each model
  """

  def __init__(self):
    pass


  def __del__(self):
    pass


  def do_log_reg(self, X, y):
    """
    do_log_reg(self): Carries out the functionality of logistic regression
    """
    # Split data into train and test
    Xtr, Xtst, ytr, ytst = train_test_split(X, y)

    print Xtr
    print Xtst
    print ytr
    print ytst

    # Train model
    log_reg = LogisticRegression()
    log_reg.fit(Xtr, ytr)
    predictions = log_reg.predict(Xtst)

    # Validate model & report error to csv

    # Test model & report error to csv

    # Report prediction & display graph



