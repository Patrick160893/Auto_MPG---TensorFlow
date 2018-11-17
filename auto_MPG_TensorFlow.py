#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 09:23:48 2018

@author: patrickorourke
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 09:37:08 2018

@author: patrickorourke
"""

# Assignment for the dataset "Auto MPG"

import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sb
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr   
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split



# Units - "Miles per galllon", "number", "Meters", "unit of power", "Newtons" . "Meters per sec sqr"
columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'car name']

learning_rate = 0.003
EPOCH = 100
# STEP 1 - GATHERING DATA

# Function to read textfile dataset and load it as a Pandas DataFrame
def loadData(file,columns):
    df = pd.read_table(file, delim_whitespace=True)
    df.columns = columns
    return df

def missingValues(dataset):
    # Identify any missing values in the dataset
    missing = dataset.isnull().sum()  
    #print("Features with missing value: ",missing)
    # Replace any missing value in the dataset with its respective column's mean 
    data.fillna(data.mean(),inplace=True)
    return data

def correlation(data):
    correlation = []
    for i in range(0,7):
        j = pearsonr(data.iloc[:,i],data.iloc[:,9])
        correlation.append(j)
    return correlation

    

    
file = "/Users/patrickorourke/Documents/Auto_MPG/auto_mpg_data_original.txt"
# Label the columsn of the Pandas DataFrame
columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'car name']
data = loadData(file,columns)
    
# STEP 2 - PREPARING THE DATA
    
# Examine the dataset
data.head()
    
data = missingValues(data)

# Create additional feature, power-to-weight ratio - higher the value, better the performance of the car
power_to_weight = data['horsepower']/data['weight']
    
#W1 = np.random.randn(7, 4)
    
#W2 = np.random.randn(4, 1)
    
# As each column in the Pandas dataframe is a Pandas Series, add the 'power to weight'column with the folowing code, using the existing indexing:
data['power to weight'] = pd.Series(power_to_weight, index=data.index)
    
train, test = train_test_split(data, test_size=0.2)
    
ys_train = np.array(train.iloc[:,9].values)
    
ys_test = np.array(test.iloc[:,9].values)

train = train.iloc[:,0:7]
    
test = test.iloc[:,0:7]

inputs = tf.placeholder(tf.float32, shape = (1,7))
labels = tf.placeholder(tf.float32, shape = (1,1))
W_1 = tf.Variable(tf.random_uniform([7,4]))
W_2 = tf.Variable(tf.random_uniform([4,1]))
b_1 = tf.Variable(tf.zeros([4]))
b_2 = tf.Variable(tf.zeros([1]))
layer_1 = tf.add(tf.matmul(inputs,W_1), b_1)
layer_1 = tf.nn.sigmoid(layer_1)
layer_2 = tf.add(tf.matmul(layer_1,W_2), b_2)
layer_2 = tf.nn.sigmoid(layer_2)

loss = tf.losses.mean_squared_error(labels = labels, predictions = layer_2)

optim = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    train_losses = []
    for e in range(EPOCH):
        
        epoch_loss = 0
        for i in range(len(train)):
            inp = train.iloc[i, :].values.reshape(1, 7)
            lab = ys_train[i].reshape(1, 1)
            iter_loss, _ = sess.run([loss, optim], feed_dict = {inputs:inp, labels:lab})
            epoch_loss += iter_loss
        
        epoch_loss /+ len(train)
        print(e, epoch_loss)
        train_losses.append(epoch_loss)

        
        
    test_loss = 0
    for i in range(len(test)):
        inp = test.iloc[i, :].values.reshape(1, 7)
        lab = ys_test[i].reshape(1, 1)
        iter_loss, _ = sess.run([loss, optim], feed_dict = {inputs:inp, labels:lab})
        test_loss += iter_loss
        
    test_loss /+ len(test)
    
    print(test_loss)
    
    plt.plot(train_losses, label='train')
    plt.xlabel("Epochs")
    plt.ylabel("Mean-Squared Loss")
    plt.title("Neural network in Tensorflow to predict 'Power-to-Weight' of automobiles", color='r')
    plt.legend()
    plt.show() 
  


    


    
       
    
    
    
    
    
    
        
        
        
        
    
        
        
        
       
        
        
    
    
    
    
    
    
    
        
        
    
        

        
        
    
        
        
    
    
   
        
    
    
    
