"""
Author: David Akre
Date: 4/15/18
Title: Market Predictor Data Controller
"""

import quandl as qd
import numpy as np
from sklearn import preprocessing

# Define String Constants
SYM_ONE    = 'SYM_ONE'
SYM_TWO    = 'SYM_TWO'
API_KEY    = 'API_KEY'
LABEL      = '_LABEL'
START_DATE = "START_DATE"

class DataController:
  """
  Class DataController : This class creates the interactions between
  the quandl API to get the market predictors data
  """

  def __init__(self, interval, forecast, config):
    qd.ApiConfig.api_key = config[API_KEY]
    self.interval = interval
    self.config   = config
    self.forecast = forecast


  def __del__(self):
    pass


  def get_data(self):
    """
    get_data(self): Is a function that will retrieve market data for
    SYM ONE and TWO
    """
    # Forward declaration
    attrs     = []
    close_one = ""
    close_two = ""

    # Fetch market data
    sym_one_data = qd.get(self.config[SYM_ONE], start_date=self.config[START_DATE], \
            collapse=self.interval) 
    sym_two_data = qd.get(self.config[SYM_TWO], start_date=self.config[START_DATE], \
            collapse=self.interval)

    # Update data frame to include user defined col attributes and drop nan elements
    for e in self.config:
      if SYM_ONE + LABEL in e:
        attrs.append(self.config[e])

    sym_one_data = sym_one_data[attrs]
    sym_one_data = sym_one_data[np.isfinite(sym_one_data[self.config[SYM_ONE+LABEL]])]

    attrs = []
    for e in self.config:
      if SYM_TWO + LABEL in e:
        attrs.append(self.config[e])

    sym_two_data = sym_two_data[attrs]
    sym_two_data = sym_two_data[np.isfinite(sym_two_data[self.config[SYM_TWO+LABEL]])]

    print sym_one_data
    print sym_two_data
    print sym_one_data.size
    print sym_two_data.size

    # Synchronize both data frames
    """
    for i, r in sym_one_data.iterrows():
      for ii, rr in sym_two_data.iterrows():
        if i is not ii:
          print i
          print ii
          sym_one_data.drop(sym_one_data.index[i])

      break

    # Create label on close data
    sym_one_data_new = sym_one_data
    sym_two_data_new = sym_two_data
    sym_one_data_new[self.config[SYM_ONE+LABEL]] = \
            sym_one_data[self.config[SYM_ONE+LABEL]].shift(-self.forecast)
    sym_two_data_new[self.config[SYM_TWO+LABEL]] = \
            sym_two_data[self.config[SYM_TWO+LABEL]].shift(-self.forecast)

    # Create X matrix for sym one and two
    X_one = np.array(sym_one_data_new.drop([self.config[SYM_ONE+LABEL]], 1))
    X_two = np.array(sym_two_data_new.drop([self.config[SYM_TWO+LABEL]], 1))

    print sym_one_data_new
    print sym_two_data_new
    print X_one
    print X_two

    # Preprocess data to standardize mean
    X_one = preprocessing.scale(X_one)
    X_two = preprocessing.scale(X_two)

    X_one_forecast = X_one[-self.forecast:]
    X_one = X_one[:-self.forecast]
    X_two_forecast = X_two[-self.forecast:]
    X_two = X_two[:-self.forecast]

    y_one = np.array(sym_one_data.drop([self.config[SYM_ONE+LABEL]]))
    y_two = np.array(sym_two_data.drop([self.config[SYM_TWO+LABEL]]))

    return X_one, X_two, y_one, y_two, sym_one_data, sym_two_data
    """


