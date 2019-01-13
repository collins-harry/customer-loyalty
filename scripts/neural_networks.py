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
if 'current_dir' in locals():
    print("Directory has been set before:")
else:
    current_dir = os.getcwd()[:-7] 
print(current_dir)
os.chdir(current_dir)
submission = pd.read_csv(current_dir + 'submissions/submission_v1.csv') #submission as template (version is not important)


# Load plain train data and test data:
data = pd.read_csv('all/train.csv')
testdata = pd.read_csv('all/test.csv')

# Load all features in the folder called 'features': (comment these lines out for custom set of features)
for f in listdir(current_dir + "\\features"):
    if (f[:4] == "test"):
        print("this is a test feature : " + f)
        temp = pd.read_csv(current_dir + "/features/" + f)
        testdata.insert(testdata.shape[1], f[13:-4], temp[f[13:-4]] )
    if (f[:5] == "train"):
        print("this is a train feature : " + f)
        temp = pd.read_csv(current_dir + "/features/" + f)
        data.insert(data.shape[1]-1, f[14:-4], temp[f[14:-4]] )
        
train = np.array(data)
np.random.shuffle(train)
trainsize = round(train.shape[0]) #fraction of trainingset used for training the model
features = train.shape[1] - 3

# A part of the training set can be used as a test set,
# so the Root Mean Squared Error can be found
train_l = train[:trainsize,features+2]
train_x = train[:trainsize,2:features+2]

test_l = train[trainsize:,features+2]
test_x = train[trainsize:,2:features+2]


# Imports for NN
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

### Baseline neural network

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