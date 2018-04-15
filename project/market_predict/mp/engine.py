""" 
Author: David Akre
Date: 4/15/18
Title: Market Predictor Engine
"""

import numpy as np
import pandas as pd
from market_predict.mp.data_controller import DataController

class Engine:
  """
  Class Engine : This class defines the core functionality for the 
  market predictor application
  """

  def __init__(self, model, interval, config, data_path):
    self.dc          = DataController(interval)
    self.model       = model
    self.config      = config
    self.data_path   = data_path


  def __del__(self):
    pass


  def predict(self):
    # Step 1 - Get data for S&P and VIX
    self.dc.get_data(self.config["API_KEY"])

    # Step 2 - Split data into train and test

    # Step 3 - Train model

    # Step 4 - Validate model & report error to csv

    # Step 5 - Test model & report error to csv

    # Step 6 - Report prediction & display graph


