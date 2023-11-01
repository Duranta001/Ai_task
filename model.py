

#import matplotlib.pyplot as plt # use for visualization
#import cv2 as cv #use for performing various function for image.
#import os # use to make connection with operating system.
import numpy as np #use for performing statistical function
import tensorflow as tf #use for deep learning algorithm.
from tensorflow import keras #use for image related deep learning work.
from tensorflow.keras import layers #use for creating deeplearning layers
from tensorflow.keras.models import Sequential #use to create sequential model  of deep learning.
#import pathlib # use to creat filepath.
import pandas as pd


x = pd.read_csv(input('Enter the training CSV dataset file path: '))
x_train = x.drop('label', axis = 1)
y_train = x['label']
x_train = np.array(x_train)
x_train = x_train.reshape(60000,28,28,1)

x_t = pd.read_csv(input('Enter the test CSV dataset file path: '))
x_test = x_t.drop('label', axis = 1)
y_test = x_t['label']
x_test = np.array(x_test)
x_test = x_test.reshape(-1,28,28,1)

x_train_scaled = x_train/255
x_test_scaled = x_test/255



num_classes = 10 #Defining the classification number.

#creating a sequential model with various layer.
model = Sequential([
     # augmentation layer
    layers.Conv2D(16,3, padding = 'same',activation='relu'), # convolutional layer feature extraction
    layers.MaxPooling2D(), #polling layer for decreasing the size of dataset.
    layers.Conv2D(32,3, padding = 'same',activation='relu'), # convolutional layer feature extraction
    layers.MaxPooling2D(), #polling layer for decreasing the size of dataset.
    layers.Conv2D(64,3, padding = 'same',activation='relu'), # convolutional layer feature extraction
    layers.MaxPooling2D(), #polling layer for decreasing the size of dataset.
    layers.Conv2D(64,3, padding = 'same',activation='relu'), # convolutional layer feature extraction
    layers.MaxPooling2D(), #polling layer for decreasing the size of dataset.
    layers.Flatten(), #flatten layer which use to convert dataset into one dimensional dataset.
    layers.Dense(3000,activation = 'relu'), # input neural network layer with 3000 neuron .
    layers.Dense(500,activation = 'relu'),  # hidden neural network layer with 500 neuron .
    layers.Dense(150, activation = 'relu'),  # hidden neural network layer with 150 neuron .
    layers.Dense(50, activation = 'relu'), # hidden neural network layer with 50 neuron .
    layers.Dense(num_classes, activation = 'sigmoid') #output layer with neuron number as no. of classification type.
    
])


model.compile(optimizer = 'adam',
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics = ['accuracy']) #compiling the model with optimizer and loss function.

result = model.fit(x_train_scaled, y_train, validation_split=0.1, batch_size =32, epochs = 50) #training the model with train dataset at 20 epoch.

model.evaluate(x_test_scaled,y_test)
