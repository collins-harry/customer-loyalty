
# coding: utf-8

import os
import datetime
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
if 'current_dir' in locals():
    print("Directory has been set before:")
else:
    current_dir = os.getcwd() 
print(current_dir)
os.chdir(current_dir)

ht = pd.read_csv(current_dir + '/Customer-Loyalty/all/historical_transactions.csv')
testpd = pd.read_csv(current_dir + '/Customer-Loyalty/all/test.csv')
testnp = np.array(testpd)

#FIND THE STARTING POINT
#indices_array = np.load('indices_array_test.npy') #000000-021938 (7134 has no transactions)
indices_array = np.zeros((testnp.shape[0],2))
for i in range(indices_array.shape[0]):
    if(indices_array[i,0] == 0):
        continue_at = i
        break
    else:
        if(testnp.shape[0]-1 == i):
            continue_at = testnp.shape[0] + 1
print("The indices of the first " + str(continue_at-1) + " card_ids in test.csv have been located in historical_transactions.csv out of the total " + str(testnp.shape[0]) + " indices.")

#FIND THE INDICES
last_found_value = continue_at
for i in range(continue_at, 100):
    testingarray = ht.index[ht.card_id == testnp[i,1]]
    indices_array[i] = testingarray[0],testingarray[testingarray.shape[0]-1]+1
    last_found_value = i
    print(i)

#SAVING THE INDICES
np.save('indices_array_test_supercomputer.npy',indices_array)
print("Last found value : " + str(last_found_value))
print("Runtime indices_array last found value (-1 / +1) : " + str(indices_array[last_found_value-1:last_found_value+2]))
print()
print("Saved indices_array last found value (-1 / +1) : " + str(np.load('indices_array_test_supercomputer.npy')[last_found_value-1:last_found_value+2]))
