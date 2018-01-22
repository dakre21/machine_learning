"""
Author: David Akre
Date: 1/21/18
Title: Half Moon Data Generator and Linear Classifier
Description: This script generates a half moon data set and
then implements a linear classifier to discriminate between
the two classes (this will also show the boundary between the two
data).
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Step 1 - Create half moon data
# NOTE: Creating noise in the data and leaving the number selection
# predictable (i.e. random_state = 0)
X, y = make_moons(n_samples=1000, noise=0.3, random_state=0)

# NOTE: Using 42 as the random number seed to get a fixed value
# size each time
Xtr, Xtst, ytr, ytst = train_test_split(X, y, random_state=42)

# Step 2 - Create 2D Grid with testI data and boundaries

# NOTE: X is the 2D sample vector returned by the make_moons function
# (e.g. [x y]).
X_arr = np.arange(X[:, 0].min(), X[:, 0].max())
y_arr = np.arange(X[:, 1].min(), X[:, 1].max())

x_grid, y_grid = np.meshgrid(X_arr, y_arr)
cm_bright = ListedColormap(['red', 'blue'])
ax = plt.subplot()
ax.set_title("Half Moon Dataset with Linear Classifier")

# Plotting testing dataset points
ax.scatter(Xtst[:, 0], Xtst[:, 1], c=ytst, cmap=cm_bright, alpha=0.8)

# Step 3 - Create Linear Classifier decision boundary (i.e. Logistic
# Regression)
log_reg = LogisticRegression()
log_reg.fit(Xtst, ytst)
Z = log_reg.decision_function(np.c_[x_grid.ravel(), y_grid.ravel()])\
        .reshape(x_grid.shape)
ax.contourf(x_grid, y_grid, Z, cmap=plt.cm.RdBu, alpha=0.6)
ax.set_xlim(x_grid.min(), x_grid.max())
ax.set_ylim(y_grid.min(), y_grid.max())

plt.show()

