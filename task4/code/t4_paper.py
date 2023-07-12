import glob, re, os
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from sklearn.model_selection import train_test_split
import time
import tensorflow as tf 

from cardiac_ml_tools import *


# data loader
data_dirs = []
regex = r'data_hearts_dd_0p2*'
DIR='../intracardiac_dataset/' # This should be the path to the intracardiac_dataset, it can be downloaded using data_science_challenge_2023/download_intracardiac_dataset.sh
for x in os.listdir(DIR):
    if re.match(regex, x):
        data_dirs.append(DIR + x)
file_pairs = read_data_dirs(data_dirs)
print('Number of file pairs: {}'.format(len(file_pairs)))
# example of file pair
print("Example of file pair:")
print("{}\n{}".format(file_pairs[0][0], file_pairs[0][1]))


# normalize data 
tic = time.time()
# Create an empty list to hold ECG and V matrices
ECG_list = [] 
V_list = []
V_list2 = []

# Loop over the cases and load the matrices
for case in range(len(file_pairs)):
    ## ECG 
    ECG = np.load(file_pairs[case][0])
    
    # Map from 10 leads to 12 
    ECG = get_standard_leads(ECG)
    
    # Normalize ECG so that the minimum and maximum for each col are separated by a distance of 1
    ECG = (ECG - ECG.min(axis=0))/(ECG.max(axis=0) - ECG.min(axis=0))

    # Append to ECG list 
    ECG_list.append(ECG)
    
    
    ## V
    V = np.load(file_pairs[case][1])
    V_list2.append(V)

    # Normalize V so that "value range is [0,1]"
    V = V - np.min(V)
    V = V/np.max(V)

    # Append the normalized V matrix to the list 
    V_list.append(V)

# Reshape into tensor for dimension agreement
ECGtens = np.array(ECG_list)
Vtens = np.array(V_list)
Vtens2 = np.array(V_list2)


toc = time.time()
print(f"time to normalize/load = {toc-tic}")

# normalize Vtens2 ("global" normalization)
Vmin = np.min(Vtens2)
Vtens2 = Vtens2 - Vmin
Vmax = np.max(Vtens2)
Vtens2 = Vtens2/Vmax

# make training and testing data
X, XX, y, yy = train_test_split(ECGtens, Vtens2, test_size=0.05, random_state=42)


class FireModule(tf.keras.layers.Layer):
    def __init__(self, squeeze_channels, expand_channels):
        super(FireModule, self).__init__()
        self.squeeze = tf.keras.layers.Conv1D(squeeze_channels, 1, activation='relu')
        self.expand1x1 = tf.keras.layers.Conv1D(expand_channels, 1, activation='relu')
        self.expand3x3 = tf.keras.layers.Conv1D(expand_channels, 3, padding='same', activation='relu')

    def call(self, x):
        squeeze = self.squeeze(x)
        expand1x1 = self.expand1x1(squeeze)
        expand3x3 = self.expand3x3(squeeze)
        return tf.concat([expand1x1, expand3x3], axis=1)



# Define model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(500,12)),

    tf.keras.layers.Conv1D(64, 3, strides=1, padding='same'),

    tf.keras.layers.MaxPool1D(3, strides=2, padding='same'),

    FireModule(16, 64),
    FireModule(16, 64), 

    tf.keras.layers.MaxPool1D(3, strides=2, padding='same'),

    FireModule(32, 128),
    FireModule(32, 128),

    tf.keras.layers.MaxPool1D(3, strides=2, padding='same'),

    FireModule(48, 192),
    FireModule(48, 192),
    FireModule(64, 256),
    FireModule(64, 256),

    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Conv1D(75, 1, strides=1),

    tf.keras.layers.AvgPool1D(),

    tf.keras.layers.Conv1D(75, 3, strides=2,padding='same'),

    tf.keras.layers.Conv1D(75, 3, strides=2,padding='same'),

    tf.keras.layers.Conv1D(75, 3, strides=2,padding='same'),

    tf.keras.layers.Conv1D(75, 3, strides=2,padding='same'),
    tf.keras.layers.ReLU()
])

# Ready the model
model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.MeanSquaredError()])

# Train the model
model.fit(X, y, epochs=30, batch_size=32)
print("training done")

# Test the model 
mse = model.evaluate(XX,yy) #gives resulting mse
print(mse)

# Mikel's metric 
def activation_time_metric(true_data, predicted_data):
    case_metrics = []
    
    for true_case, predicted_case in zip(true_data, predicted_data):
        true_act = []
        pred_act = []
        for node in range(0,75,1):
            true_act.append(np.argmax(true_case[:,node]>0)
            pred_act.append(np.argmax(predicted_case[:,node]>0)
        
        abs_diff = np.abs(np.array(true_act) - np.array(pred_act))
        case_metric = np.mean(abs_diff) #, np.var(abs_diff)
        case_metrics.append(case_metric)
    
    avg_metrics = np.mean(case_metrics, axis=0)
    var_metrics = np.var(case_metrics, axis=0)
    
    return avg_metrics, var_metrics


ypred = model.predict(XX)
avg, var = activation_time_metric(yy,ypred)
print("Activation Time Absolute Difference Average:", avg)
print("Activation Time Absolute Difference Variance:", var)

