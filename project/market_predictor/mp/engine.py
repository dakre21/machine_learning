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
    self.interval    = interval
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
    plt.figure(figsize=(15,10))
    plt.suptitle("Market Predictor Results")

    ax = plt.subplot(2, 2, 1)
    ax.set_title("Original " + self.dc.config[SYM_ONE])
    ax.set_xlabel("Epochs reported %s since start date %s" % (self.interval, \
            self.dc.config[START_DATE]))
    ax.set_ylabel("Future price ($)")
    ax.plot(X[:,0], X[:,1], color='blue')

    ax = plt.subplot(2, 2, 2)
    ax.set_title("Original " + self.dc.config[SYM_TWO])
    ax.set_xlabel("Epochs reported %s since start date %s" % (self.interval, \
            self.dc.config[START_DATE]))
    ax.set_ylabel("Future price ($)")
    ax.plot(X[:,0], X[:,2], color='green')


  def _plot(self, X, Xtr, Xtst, X_fc, ytr, pred, forecast):
    """
    _plot(self, Xtst, X_fc, pred, forecast): is a private function that
    will plot the data post learning and test
    """
    ax = plt.subplot(2, 2, 3)
    ax.set_title("Training Results")
    ax.set_xlabel("Epochs reported %s since start date %s" % (self.interval, \
            self.dc.config[START_DATE]))
    ax.set_ylabel("Future price ($)")
    ax.scatter(Xtr[:,0], ytr, color='red', s=0.75)
    ax.plot(X[:,0], X[:,2], color='orange')


    ax = plt.subplot(2, 2, 4)
    ax.set_title("Testing and Forecast Results")
    ax.set_xlabel("Epochs reported %s since start date %s" % (self.interval, \
            self.dc.config[START_DATE]))
    ax.set_ylabel("Future price ($)")
    ax.scatter(Xtst[:,0], pred, color='red', s=0.75)
    ax.scatter(X_fc[-self.dc.forecast:,0], forecast[-self.dc.forecast:], color='violet', s=0.75)
    ax.plot(X[:,0], X[:,2], color='orange')

    plt.show()


  def predict(self):
    """
    predict(self): Core function that carries out the logic for the entire 
    application
    """

    if self.model == LIN_REG:
      X, y, X_fc, X_plt = self.dc.get_data_sklearn()
      self._init_plot(X_plt)
      Xtr, Xtst, X_fc, ytr, pred, forecast = self.mc.do_sklearn(X, y, X_fc, self.model)
      self._plot(X_plt, Xtr, Xtst, X_fc, ytr, pred, forecast)   

    elif self.model == ARIMA:
      # TODO - Implement
      pass
    else:
      # TODO - Implement
      pass

    
