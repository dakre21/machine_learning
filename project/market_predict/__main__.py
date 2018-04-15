"""
Author: David Akre
Date: 4/15/18
Title: Market Predictor Application main
"""

import click
from mp import engine
from mp.validate import validate_inputs

# Global declarations
models = [
  "SVM",
  "Naive Bayes",
  "KNN",
  "Logistic Regression",
  "Neural Network"
]

intervals = [
  "day",
  "week",
  "month",
  "year"
]

@click.command()
@click.option('-m', '--model', help='Available options are: '
        + str(models), required=True)
@click.option('-t', '--interval', help='Available options are: '
        + str(intervals), required=True)
def main(model, interval):
  """
  main() : Main entry point for the market predictor application.
  User must provide a model and an appropriate timing interval
  to the application.
  """

  if validate_inputs(model, interval, models, intervals) == False:
    print "Error in the inputs, please run market_predict --help"
    return

  eng = Engine(model, interval)
  eng.predict()


if __name__ == "__main__":
  main()

