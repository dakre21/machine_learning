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
    ax.plot(X[:,0], X[:,1], color='red')

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
    ax.set_title("Training Results with Model %s" % self.model)
    ax.set_xlabel("Epochs reported %s since start date %s" % (self.interval, \
            self.dc.config[START_DATE]))
    ax.set_ylabel("Future price ($)")
    ax.scatter(Xtr[:,0], ytr, color='blue', s=4)
    ax.plot(X[:,0], X[:,2], color='green')

    ax = plt.subplot(2, 2, 4)
    ax.set_title("Testing and Forecast Results with Model %s" % self.model)
    ax.set_xlabel("Epochs reported %s since start date %s" % (self.interval, \
            self.dc.config[START_DATE]))
    ax.set_ylabel("Future price ($)")
    ax.scatter(Xtst[:,0], pred, color='blue', s=4)
    fn = np.polyfit(X_fc[-self.dc.forecast:,0], forecast[-self.dc.forecast:], 3)
    fn = np.poly1d(fn)
    ax.plot(X_fc[-self.dc.forecast:,0], fn(X_fc[-self.dc.forecast:,0]), '-', \
            color='violet')
    ax.scatter(X_fc[-self.dc.forecast:,0], forecast[-self.dc.forecast:], \
            color='orange', s=1)
    ax.plot(X[:,0], X[:,2], color='green')

    plt.show()


  def predict(self):
    """
    predict(self): Core function that carries out the logic for the entire 
    application
    """

    # Get data for sklearn & setup initial plot
    X, y, X_fc, X_plt = self.dc.get_data_sklearn()
    self._init_plot(X_plt)

    # Train and test then plot results
    Xtr, Xtst, ytr, pred, forecast = self.mc.do_sklearn(X, y, X_fc, self.model)
    self._plot(X_plt, Xtr, Xtst, X_fc, ytr, pred, forecast)   
   

