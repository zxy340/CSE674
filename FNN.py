import os
from sklearn.preprocessing import normalize
import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense
from scipy.io import loadmat
from dataloader import data_loader

# load data from csv
data_train, label_train, data_val, label_val = data_loader()

#network size
input_dim = 11
hidden_dim = 128

net = Sequential([
  Dense(hidden_dim, activation='relu', use_bias=False, input_shape=(input_dim,)),  # hidden layer
  Dense(2, activation='softmax', use_bias=False, name='Dense_1'),
  Dense(2, activation='softmax', use_bias=False, name='Dense_2'),  # output layer
])

# loss is cross entropy
net.compile(
  loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'],
)

net.fit(
  data_train,
  label_train,
  epochs=20,
  batch_size=32,
  validation_data=(data_val, label_val)
)

net.save('FNN.h5')