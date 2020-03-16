#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 17:05:06 2020

@author: leonardo
"""

# libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing dataset
dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoderX = LabelEncoder()
X[:, 0] = labelEncoderX.fit_transform(X[:, 0])
oneHotEncoder = OneHotEncoder(categorical_features = [0])
X = oneHotEncoder.fit_transform(X).toarray()

labelEncoderY = LabelEncoder()
y = labelEncoderY.fit_transform(y)

# splitting the datase into the training set and test set
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, Ytest = train_test_split(X, y, test_size = 0.2, random_state = 0)

# feature scalling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
xTrain = sc_X.fit_transform(xTrain)
xTest = sc_X.transform(xTest)"""
