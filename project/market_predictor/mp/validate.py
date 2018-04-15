""" 
Author: David Akre
Date: 4/15/18
Title: Market Predictor Validator
"""


def validate_inputs(model, interval, models, intervals):
  """
  validate_inputs(model, interval) : Validates the user inputs from click
  """
  
  # Forward declaration
  rc = False

  for m in models:
    if m == model:
      rc = True
      break

  for i in intervals:
    if i == interval:
      rc &= True
      break

  return rc

