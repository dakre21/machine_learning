""" 
Author: David Akre
Date: 4/15/18
Title: Market Predictor Model Controller
"""

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
    X = preprocessing.scale(X)

    # Split data into train and test
    Xtr, Xtst, ytr, ytst = train_test_split(X, y)

    # Train model
    if model == LIN_REG: 
      clf = LinearRegression()

    clf.fit(Xtr, ytr)
    predictions = clf.predict(Xtst)
    confidence = clf.score(Xtst, ytst)
    print X_fc
    forecast = clf.predict(X_fc)
    print predictions
    print "CONFIDENCE"
    print confidence
    print "FORECAST"
    print forecast

    # Validate model & report error to csv

    # Test model & report error to csv

    # Report prediction & display graph



