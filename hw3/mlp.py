"""
Author: David Akre
Date: 3/3/18
Title: Multi-Layer Perceptron
Description: Train a neural netrok on the classical MNIST data set
where the implementation of the MLP will contain at least two different
hidden layer sizes and utilizes regularization. The program will then
report the classification error on the training and testing data sets 
for each configuration.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


if __name__ == "__main__":
  tf.app.run()
