import scipy.io as sio
import scipy as scipy
import numpy as np
import os
from concise.preprocessing import encodeDNA
from keras import backend as K
import matplotlib.pyplot as plt


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name
        
def r2_score_k(y_true, y_pred):  
    total_error = K.sum(K.square(y_true- K.mean(y_true)))
    unexplained_error = K.sum(K.square(y_true - y_pred))
    return 1 - unexplained_error/total_error

def get_data(X, Y):
    data =  ({'seq': X},Y)
    for i in range(X.shape[1]):
        data[0]['seq' + str(i)] = X[:,i]      
    
    return data[0], data[1]