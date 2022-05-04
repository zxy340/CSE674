import os
from sklearn.preprocessing import normalize
import numpy as np
import keras
from keras.models import Sequential
from keras.layers.core import Dense
from dataloader import data_loader, data_loader_CNNimage, data_loader_meanimage

# load MLP model
model = keras.models.load_model('CNN.h5')

# ...................... custom parameter ............................
# data_train, label_train, data_val, label_val = data_loader_CNNimage(CNN_model=model)  # choose to use the predicted result as the feature
data_train, label_train, data_val, label_val = data_loader_meanimage()  # choose to use the mean of the image as the feature
input_dim = 12  # 10 when choose CNNimage, 12 when choose meanimage
# ....................................................................

label_train = np.eye(2)[label_train.reshape(-1)]
label_val = np.eye(2)[label_val.reshape(-1)]

net = Sequential([
  Dense(128, activation='relu', use_bias=False, input_dim=input_dim),  # hidden layer
  Dense(2, activation='softmax', use_bias=False, name='Dense_1'),
  Dense(2, activation='softmax', use_bias=False, name='Dense_2'),  # output layer
])

# loss is cross entropy
net.compile(
  loss='binary_crossentropy', optimizer='adam', metrics=['AUC'],
)

net.fit(
  data_train,
  label_train,
  epochs=20,
  batch_size=32,
  validation_data=(data_val, label_val)
)