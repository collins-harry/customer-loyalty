import os
import datetime
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def main():
    load_data()

def load_data(features = 'all', custom_features = [], shuffle = True):
    '''
    Returns: train_x, train_l, testset_x

    Parameters
    ---------
    features -- 'all' : use all features including custom features
                'original' : use only original features
                'custom' : use original features + specified custom features 
    custom_features -- names of custom feature csv files
    shuffle -- shuffle training set

    '''

    ###deciding on directory structure##########
    if 'current_dir' in locals():
        print("Directory has been set before:")
    else:
        current_dir = os.path.dirname(os.path.realpath(__file__))
    print(current_dir)
    os.chdir(current_dir)
    submission = pd.read_csv(current_dir + '/../submissions/submission_v1.csv') #submission as template (version is not important)
    ############################################

    data = pd.read_csv(current_dir+'/../all/train.csv')
    testdata = pd.read_csv(current_dir+ '/../all/test.csv')
    
    data, testdata = add_features(features, custom_features, data, testdata, current_dir)

    training_set = np.array(data)
    features = training_set.shape[1] - 3

    if shuffle == True:
        np.random.shuffle(training_set)

    train_l = training_set[:,features+2]
    train_x = training_set[:,2:features+2]

    testset = np.array(testdata)
    testset_x = testset[:,2:features+2]

    return train_x, train_l, testset_x


def add_features(features, custom_features, data, testdata, current_dir):

    if features == 'original':

        return data, testdata


    elif features == 'all':

        for f in listdir(current_dir + "/../features"):
            if (f[:4] == "test"):
                print("this is a test feature : " + f)
                temp = pd.read_csv(current_dir + "/../features/" + f)
                testdata.insert(testdata.shape[1], f[13:-4], temp[f[13:-4]] )
            if (f[:5] == "train"):
                print("this is a train feature : " + f)
                temp = pd.read_csv(current_dir + "/../features/" + f)
                data.insert(data.shape[1]-1, f[14:-4], temp[f[14:-4]] )
        return data, testdata

    
    elif features == 'custom':
        # Load a custom set of features: (comment these lines out for all features)

        for f in custom_features:
            if (f[:4] == "test"):
                print("this is a test feature : " + f)
                temp = pd.read_csv(current_dir + "/../features/" + f)
                testdata.insert(testdata.shape[1], f[13:-4], temp[f[13:-4]] )
            if (f[:5] == "train"):
                print("this is a train feature : " + f)
                temp = pd.read_csv(current_dir + "/../features/" + f)
                data.insert(data.shape[1]-1, f[14:-4], temp[f[14:-4]] )
        return data, testdata

if __name__ == '__main__':
    main()
