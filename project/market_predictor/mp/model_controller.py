""" 
Author: David Akre
Date: 4/15/18
Title: Market Predictor Model Controller
"""

import numpy as np
from market_predictor import *
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression


class ModelController:
  """
  Class ModelController :  Class defines the functionality carried out by each model
  """

  def __init__(self):
    pass


  def __del__(self):
    pass


  def do_sklearn(self, X, y, X_fc, model):
    """
    do_sklearn(self, X, y, X_fc, model): Carries out the functionality of sklearn
    """
    # Forward declarations
    clf = None

    # Preprocess data
    imp = preprocessing.Imputer()
    X = preprocessing.scale(X)

    # Split data into train and test
    Xtr, Xtst, ytr, ytst = train_test_split(X, y)

    # Train model
    if model == LIN_REG: 
      clf = LinearRegression()

    clf.fit(Xtr, ytr)
    predictions = clf.predict(Xtst)
    confidence = clf.score(Xtst, ytst)

    # Forecast data now
    X_fc = np.vstack((X, X_fc))
    X_fc = imp.fit_transform(X_fc)
    forecast = clf.predict(X_fc)


