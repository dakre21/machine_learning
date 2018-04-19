"""
Author: David Akre
Date: 4/15/18
Title: Market Predictor Data Controller
"""

import quandl as qd
import numpy as np
import pandas as pd
from market_predictor import *
from datetime import datetime, timedelta
from sklearn import preprocessing

# Define String Constants
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


  def _convert_dates_to_int(self, dates):
    """
    _convert_dates_to_int(self, dates) takes a well formated TimeStamp date formart
    and convert it into a list of ints which sklearn and other like libraries can 
    process for learning
    """

    # Maintain a good history of the real dates as a class object
    self.dates = dates
    dates = [x for x in range(len(dates))]

    return dates


  def _forecast_data(self):
    """
    _forecast(self, vals_one, vals_two, dates) is a private function which extends 
    the dataframe out to the forcasted timeline
    """
    # Forward declarations
    one   = []
    two   = []
    dates = []

    # TODO: If choosing weekly, quarterly, annually etc fix this to extend by that amount
    tomorrow = datetime.now() + timedelta(1)
    tmp = np.array(pd.date_range(tomorrow, periods=self.forecast))

    tmp = [pd.Timestamp(x) for x in tmp]
    dates = np.append(dates, tmp)

    fc = np.append(one, np.repeat(np.nan, self.forecast))

    return fc, dates


  def get_data_sklearn(self):
    """
    get_data_log_reg(self): Is a function that will fetch data from quandl
    and forecast it for sklearn logistic regression
    """
    vals_one, vals_two, dates = self._get_data() 
   
    # Setup data for training and testing
    dates = self._convert_dates_to_int(dates)
    X_one = np.column_stack((dates, vals_one))

    # Setup data for plot
    # TODO: Change this around and refactor dates to plot has actual datetimes
    X_plt = np.column_stack((dates, vals_one, vals_two))

    # Setup forecasting data
    fc, dates = self._forecast_data()
    dates = self._convert_dates_to_int(dates)
    X_fc = np.column_stack((dates, fc))

    return X_one, vals_two, X_fc, X_plt


  def _get_data(self):
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


    return vals_one, vals_two, dates_one

