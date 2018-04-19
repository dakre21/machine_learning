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
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import LassoLars
from sklearn.ensemble import BaggingRegressor


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
    clf     = None

    # Preprocess data
    imp = preprocessing.Imputer()

    # Split data into train and test
    Xtr, Xtst, ytr, ytst = train_test_split(X, y)

    # Train model - Assumption default settings
    if model == LIN_REG: 
      clf = LinearRegression()
    elif model == BAGGING:
      clf = BaggingRegressor()
    elif model == SGD:
      clf = SGDRegressor()
    elif model == PAL:
      clf = PassiveAggressiveRegressor()
    else:
      clf = LassoLars()

    clf.fit(Xtr, ytr)
    predictions = clf.predict(Xtst)
    accuracy = clf.score(Xtst, ytst)

    # Forecast data now
    X_fc = np.vstack((X, X_fc))
    X_fc_og = X_fc
    X_fc = imp.fit_transform(X_fc)
    forecast = clf.predict(X_fc)

    # Report Accuracy for Model
    logger.info("The accuracy of the %s model was %0.4f" % (model, accuracy))

    return Xtr, Xtst, X_fc_og, ytr, predictions, forecast


