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
from sklearn.model_selection import train_test_split

# Setup verbose logging with info set
tf.logging.set_verbosity(tf.logging.INFO)

def back_prop_noreg_fn(features, labels, mode):
  """
  back_prop_noreg_fn : model function for back propagation algorithm
  neural network w/ 2 hidden layers and no regularization
  """
  # Reshape input data to 4-D tensor
  # According to Tensorflow's documentation on MNIST data the following applies:
  # MNIST data are 28x28 pixels and have 1 color channels
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # 1st hidden layer - apply convolution to first layer
  conv = tf.layers.conv2d(
    inputs=input_layer,
    filters=32,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu
  )

  # 2nd hidden layer - apply batch normalization
  batch = tf.layers.batch_normalization(
    inputs=conv
  )

  # Flatten batch data set and apply it to dense layer
  size = batch.get_shape().as_list()
  batch_flat = tf.reshape(batch, [-1, size[0] * size[1] * size[2] * size[3]])
  dense = tf.layers.dense(inputs=batch, units=1024, activation=tf.nn.relu)

  # Generate predictions on mnist data
  predictions = {
    "classes" : tf.argmax(input=dense, axis=1),
    "probabilities" : tf.nn.softmax(dense, name="softmax_tensor")
  }

  # Check if the mode is equal to the PREDICT mode key
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate the loss of training and evaluations
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=dense)

  # Apply SGB to training data if in training mode
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(
      loss=loss,
      global_step=tf.train.get_global_step()
    )
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Return the evaluation metrics
  eval_metric_ops = {
    "accuracy" : tf.metrics.accuracy(
      labels=labels,
      predictions=predictions["classes"]
    )
  }

  return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


  
def back_prop_reg_fn(features, labels, mode):
  """
  back_prop_reg_fn : model function for back propagation algorithm
  neural network w/ 2 hidden layers and regularization
  """
  # Reshape input data to 4-D tensor
  # According to Tensorflow's documentation on MNIST data the following applies:
  # MNIST data are 28x28 pixels and have 1 color channels
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # Setup l2 regularization
  regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

  # 1st hidden layer - apply convolution to first layer
  conv = tf.layers.conv2d(
    inputs=input_layer,
    filters=32,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu,
    kernel_regularizer=regularizer
  )

  # 2nd hidden layer - apply batch normalization
  batch = tf.layers.batch_normalization(
    inputs=conv,
    kernel_regularizer=regularizer
  )

  # Flatten batch data set and apply it to dense layer
  size = batch.get_shape().as_list()
  batch_flat = tf.reshape(batch, [-1, size[0] * size[1] * size[2] * size[3]])
  dense = tf.layers.dense(inputs=batch, units=1024, activation=tf.nn.relu)

  # Generate predictions on mnist data
  predictions = {
    "classes" : tf.argmax(input=dense, axis=1),
    "probabilities" : tf.nn.softmax(dense, name="softmax_tensor")
  }

  # Check if the mode is equal to the PREDICT mode key
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate the loss of training and evaluations
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=dense)

  # Apply SGB to training data if in training mode
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(
      loss=loss,
      global_step=tf.train.get_global_step()
    )
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Return the evaluation metrics
  eval_metric_ops = {
    "accuracy" : tf.metrics.accuracy(
      labels=labels,
      predictions=predictions["classes"]
    )
  }

  return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)



def main(argv):
  """
  main : main entry point functino for tensorflow
  """
  # Get the training and evaluation data
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data = mnist.test.images
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

  # Create an estimator w/o regularization
  mnist_clf_noreg = tf.estimator.Estimator(model_fn=back_prop_noreg_fn, 
    model_dir="mnist_backprop_noreg")

  # Create an estimator w/ regularization (l2)
  mnist_clf_reg = tf.estimator.Estimator(model_fn=back_prop_reg_fn,
    model_dir="mnist_backprop_reg")
  
  # Setup training input function
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True
  )

  # Setup eval input function
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},
    y=eval_labels,
    num_epochs=None,
    shuffle=False
  )

  # Train the model w/o regularization & rounds = 50
  mnist_clf_noreg.train(
    input_fn=train_input_fn,
    steps=50,
    hooks=None
  )
  eval_results_noreg_50 = mnist_clf_noreg.evaluate(input_fn=eval_input_fn)

  # Train the model w/o regularization & rounds = 250
  mnist_clf_noreg.train(
    input_fn=train_input_fn,
    steps=250,
    hooks=None
  )
  eval_results_noreg_250 = mnist_clf_noreg.evaluate(input_fn=eval_input_fn)

  # Train the model w/ regularization & rounds = 50
  mnist_clf_reg.train(
    input_fn=train_input_fn,
    steps=50,
    hooks=None
  )
  eval_results_reg_50 = mnist_clf_reg.evaluate(input_fn=eval_input_fn)

  # Train the model w/ regularization & rounds = 250
  mnist_clf_reg.train(
    input_fn=train_input_fn,
    steps=250,
    hooks=None
  )
  eval_results_reg_250 = mnist_clf_reg.evaluate(input_fn=eval_input_fn)
  
  print(eval_results_noreg_50)
  print(eval_results_noreg_250)
  print(eval_results_reg_50)
  print(eval_results_reg_250)


if __name__ == "__main__":
  tf.app.run()
