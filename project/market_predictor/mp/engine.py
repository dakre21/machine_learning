""" 
Author: David Akre
Date: 4/15/18
Title: Market Predictor Engine
"""

import numpy as np
import pandas as pd
from market_predictor.mp.data_controller import DataController
from sklearn.model_selection import train_test_split

class Engine:
  """
  Class Engine : This class defines the core functionality for the 
  market predictor application
  """

  def __init__(self, model, interval, forecast, config, data_path):
    self.dc = DataController(interval, forecast, config)
    self.model       = model
    self.data_path   = data_path


  def __del__(self):
    pass


  def predict(self):
    # Step 1 - Get data for S&P and VIX
    X, y = self.dc.get_data()

    # Step 2 - Split data into train and test
    Xtr, Xtst, ytr, ytst = train_test_split(X, y, random_state=42)

    print Xtr
    print Xtst
    print ytr
    print ytst

    # Step 3 - Train model

    # Step 4 - Validate model & report error to csv

    # Step 5 - Test model & report error to csv

    # Step 6 - Report prediction & display graph


