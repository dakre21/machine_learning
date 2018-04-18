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


  def _forecast(self, X, nc, y=None):
    """
    _forecast(self, X, nc, y=None): Is a private function that will extend the
    data frame to the forecasted amount
    """


  def get_data(self):
    """
    get_data(self): Is a function that will retrieve market data for
    SYM ONE and TWO
    """
    # Forward declaration
    close_one = ""
    close_two = ""
    count     = 0

    # Couple of globals
    global dates_one, dates_two, vals_one, vals_two

    # Fetch market data
    sym_one_data = qd.get(self.config[SYM_ONE], start_date=self.config[START_DATE], \
            collapse=self.interval) 
    sym_two_data = qd.get(self.config[SYM_TWO], start_date=self.config[START_DATE], \
            collapse=self.interval)
    sym_one_data = sym_one_data[self.config[SYM_ONE+LABEL]]
    sym_two_data = sym_two_data[self.config[SYM_TWO+LABEL]]

    # Synchronize data
    dates_one = np.array(sym_one_data.index.tolist())
    dates_two = np.array(sym_two_data.index.tolist())
    vals_one  = np.array(sym_one_data.values.tolist())
    vals_two  = np.array(sym_two_data.values.tolist())
 
    dates = dates_one
    for d_one in dates:
      if d_one not in dates_two:
        dates_one = np.delete(dates_one, count)
        vals_one = np.delete(vals_one, count)
      else:
        count += 1

    count = 0
    dates = dates_two
    for d_two in dates:
      if d_two not in dates_one:
        dates_two = np.delete(dates_two, count)
        vals_two = np.delete(vals_two, count)
      else:
        count += 1

    count = 0
    vals = vals_one
    for v_one in vals:
      if np.isnan(v_one):
        vals_one = np.delete(vals_one, count)
        dates_one = np.delete(dates_one, count)
        vals_two  = np.delete(vals_two, count)
        dates_two = np.delete(dates_two, count)
      else:
        count += 1

    count = 0
    vals = vals_two
    for v_two in vals:
      if np.isnan(v_two):
        vals_two = np.delete(vals_two, count)
        dates_two = np.delete(dates_two, count)
        vals_one = np.delete(vals_one, count)
        dates_one = np.delete(dates_one, count)
      else:
        count += 1

    X_one = np.vstack((dates_one, vals_one)).T
    X_two = np.vstack((dates_two, vals_two)).T

    print X_one
    print X_two

    #sym_two_data = sym_two_data[np.isfinite(sym_two_data[self.config[SYM_TWO+LABEL]])]
    #sym_one_data = sym_one_data[np.isfinite(sym_one_data[self.config[SYM_ONE+LABEL]])]

    """
    r, c = X_one.shape
    rr, cc = X_two.shape

    print X_one
    print X_two

    print one
    print two

    if r != rr:
      logger.error("The data sizes from each symbol do not match sym_one %d != sym_two %d" \
              %r %rr)
      return 

    self._forecast(X_one, r)
    """


