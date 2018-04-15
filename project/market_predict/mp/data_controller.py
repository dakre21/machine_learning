"""
Author: David Akre
Date: 4/15/18
Title: Market Predictor Data Controller
"""


class DataController:
  """
  Class DataController : This class creates the interactions between
  the quandl API to get the market predictors data
  """

  def __init__(self, interval):
    self.vix_data = None
    self.sp_data  = None


  def __del__(self):
    pass


  def get_data(self, api_key):
    """
    get_data(self): Is a function that will retrieve market data for
    the VIX and S&P500 index for a specific interval
    """
    print api_key
