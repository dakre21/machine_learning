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

  def __init__(self, interval, config):
    qd.ApiConfig.api_key = config['API_KEY']
    self.interval = interval
    self.config   = config


  def __del__(self):
    pass


  def get_data(self):
    """
    get_data(self): Is a function that will retrieve market data for
    SYM ONE and TWO
    """
    # Forward declaration
    attrs = []

    # Fetch market data
    sym_one_data = qd.get(self.config['SYM_ONE'], collapse=self.interval) 
    sym_two_data = qd.get(self.config['SYM_TWO'], collapse=self.interval)

    # Update data frame to include user defined col attributes
    for e in self.config:
      if "SYM_ONE_ATTR" in e:
        attrs.append(self.config[e])

    sym_one_data = sym_one_data[attrs]

    attrs = []
    for e in self.config:
      if "SYM_TWO_ATTR" in e:
        attrs.append(self.config[e])

    sym_two_data = sym_two_data[attrs]

    print sym_one_data
    print sym_two_data


