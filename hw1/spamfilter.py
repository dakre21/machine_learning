"""
Author: David Akre
Date: 1/21/18
Title: Naive Bayes Spam Filter
Description: Implement a Naive Bayes filter and report the 5-Fold
Validation error based on the input spam filter data set.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.naive_bayes import GaussianNB

# Step 0: Forward declarations
filepath = "data/spambase_train.csv"
nf = 5

# Step 1: Read in spamfilter training data
dataset = pd.read_csv(filepath)
X = dataset.values[:,:-1]
y = dataset.values[:,-1]
size = dataset.columns.size

# Step 2: Create 5-Fold Validation Model and Report Error
kf = KFold(n_splits=nf)
kf = kf.split(X)

# Step 3: Create Naive Bayes Filter (using GaussianNB library call from
# sklearn)
score = []
for test, train in kf:
    gnb = GaussianNB()
    score.append(cross_val_score(gnb, X[test], y[test]))

# Step 4: Compute validation output
output = 0
i = 0
for si in score:
    for sn in si:
        output += sn
        i += 1

avg = output / i

print "5-Fold Validation Error Output: %f" % (avg)


