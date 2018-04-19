""" 
Author: David Akre
Date: 4/15/18
Title: Market Predictor Engine
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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
    self.cm_bright   = ListedColormap(['red', 'blue'])


  def __del__(self):
    pass


  def _init_plot(self, X):
    """
    init_plot(self, X, y) is a private function that initializes the plot 
    scheme for this application
    """
    plt.figure(figsize=(20,10))
    ax = plt.subplot(2, 2, 1)
    ax.set_title("Original " + self.dc.config[SYM_ONE] + " & " + self.dc.config[SYM_TWO])
    ax.scatter(X[:,0], X[:,1], cmap=self.cm_bright, alpha=0.8)
    plt.show()


  def predict(self):
    """
    predict(self): Core function that carries out the logic for the entire 
    application
    """

    if self.model == LIN_REG:
      X, y, X_fc, X_plt = self.dc.get_data_sklearn()
      self._init_plot(X_plt)
      self.mc.do_sklearn(X, y, X_fc, self.model)
    elif self.model == ARIMA:
      # TODO - Implement
      pass
    else:
      # TODO - Implement
      pass

    
