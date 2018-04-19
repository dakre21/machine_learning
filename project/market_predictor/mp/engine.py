""" 
Author: David Akre
Date: 4/15/18
Title: Market Predictor Engine
"""

import numpy as np
import pandas as pd
from market_predictor import *
from market_predictor.mp.data_controller import DataController
from market_predictor.mp.model_controller import ModelController


class Engine:
  """
  Class Engine : This class defines the core functionality for the 
  market predictor application
  """

  def __init__(self, model, interval, forecast, config, data_path):
    self.dc          = DataController(interval, forecast, config)
    self.mc          = ModelController() 
    self.model       = model
    self.data_path   = data_path


  def __del__(self):
    pass


  def predict(self):
    """
    predict(self): Core function that carries out the logic for the entire 
    application
    """

    if self.model == LIN_REG:
      X, y, X_fc = self.dc.get_data_sklearn()
      self.mc.do_sklearn(X, y, X_fc, self.model)
    elif self.model == ARIMA:
      # TODO - Implement
      pass
    else:
      # TODO - Implement
      pass

    
