"""
Author: David Akre
Date: 4/15/18
Title: Market Predictor Data Controller
"""

import quandl as qd
import numpy as np

class DataController:
  """
  Class DataController : This class creates the interactions between
  the quandl API to get the market predictors data
  """

  def __init__(self, interval, api_key, sp_sym, vix_sym):
    qd.ApiConfig.api_key = api_key
    self.interval = interval
    self.vix_sym  = vix_sym
    self.sp_sym   = sp_sym
    self.vix_data = np.array()
    self.sp_data  = np.array()


  def __del__(self):
    pass


  def get_data(self):
    """
    get_data(self): Is a function that will retrieve market data for
    the VIX and S&P500 index for a specific interval
    """
    self.sp_data = qd.get(self.sp_sym, collapse=self.interval, 
            returns="numpy")
    self.vix_data = qd.get(self.vix_sym, collapse=self.interval, 
            returns="numpy")


