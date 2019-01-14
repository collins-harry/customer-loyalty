# -*- coding: utf-8 -*-
"""
NEURAL NETWORK MODELS FOR CUSTOMER-LOYALTY
"""

import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import preprocessing

# Imports for NN
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

### Preprocessing
train_x, train_l, testset_x = preprocessing.load_data()

# define base model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(13, input_dim=features, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, epochs=1, batch_size=5, verbose=2)
kfold = KFold(n_splits=2, random_state=seed)
results = cross_val_score(estimator, train_x, train_l, cv=kfold)
history_callback = estimator.fit(train_x, train_l)
loss_history = history_callback.history["loss"]
numpy_loss_history = np.array(loss_history)
np.savetxt("loss_history.txt", numpy_loss_history, delimiter=",")
#a = estimator.callbacks.Callback()
#print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))


#
#with open('baseline_model.txt', 'w') as f:
#    estimator = KerasRegressor(build_fn=baseline_model, epochs=1, batch_size=5, verbose=2)
#    kfold = KFold(n_splits=2, random_state=seed)
#    results = cross_val_score(estimator, train_x, train_l, cv=kfold)
#    print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
