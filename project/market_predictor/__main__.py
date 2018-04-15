"""
Author: David Akre
Date: 4/15/18
Title: Market Predictor Application main
"""

import click
import yaml
from mp.engine import Engine
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


def _read_config(config_path):
  """
  _read_config(config_path) is a helper function that will read in the yaml
  config file to program memory
  """

  # Forward declarations
  rc = True
  global config

  with open(config_path, 'r') as data:
    try:
      config = yaml.load(data)
    except yaml.YAMLError as exc:
      rc = False
      print exc

  return rc


@click.command()
@click.option('-m', '--model', help='Available options are: '
        + str(models), required=True)
@click.option('-t', '--interval', help='Available options are: '
        + str(intervals), required=True)
@click.option('-c', '--config_path', help='Provide a path to the config file',
        required=True)
@click.option('-d', '--data_path', help='Provide a path to a data directory',
        required=False)
def main(model, interval, config_path, data_path):
  """
  main() : Main entry point for the market predictor application.
  User must provide a model, an appropriate timing interval, and env path
  to the application.
  """

  # First validate core user arguments
  if validate_inputs(model, interval, models, intervals) != True:
    print "Error in the inputs, please run market_predictor --help"
    return

  # Second read configuration file
  if _read_config(config_path) != True:
    print "Error while reading the configuration yaml file. Please provide "\
            "a valid one... follow example_config.yaml in project"
    return 

  eng = Engine(model, interval, config, data_path)
  eng.predict()


if __name__ == "__main__":
  main()

