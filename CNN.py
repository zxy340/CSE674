import tensorflow as tf
import numpy as np
from tensorflow import keras
import os
from numpy.random import seed
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from keras import regularizers
import matplotlib.pyplot as plt
from keras.optimizers import adam_v2
from sklearn.preprocessing import LabelEncoder

# load data from csv
df1 = pd.read_csv(r'.\csv\R6_yes.csv')
df2 = pd.read_csv(r'.\csv\R6_no.csv')
df3 = pd.read_csv(r'.\csv\R7_yes.csv')
df4 = pd.read_csv(r'.\csv\R7_no.csv')
df5 = pd.read_csv(r'.\csv\R8_yes.csv')
df6 = pd.read_csv(r'.\csv\R8_no.csv')
df7 = pd.read_csv(r'.\csv\R9_yes.csv')
df8 = pd.read_csv(r'.\csv\R9_no.csv')
df9 = pd.read_csv(r'.\csv\R10_yes.csv')
df10 = pd.read_csv(r'.\csv\R10_no.csv')
df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10], axis=0)

#reindex of all data after concat
df.reset_index(drop=True, inplace=True)

allimg = []
for i in tqdm(range(df.shape[0])):
    img = image.load_img(r'./crop/' + str(df['imgspid'][i]) + '.jpg',
                         target_size=(224, 224, 3))
    img = image.img_to_array(img)
    img = img / 255
    allimg.append(img)

# 转换为numpy数组
img_total = np.array(allimg)

# for softmax
encoder = LabelEncoder()
AD_values = encoder.fit_transform(df['dementia'].values)
AD_values = np.array([AD_values]).T

# for softmax
enc = OneHotEncoder()
y_total = enc.fit_transform(AD_values)
y_total = y_total.toarray()

seed(2)
tf.random.set_seed(9)

X_train, X_test, y_train, y_test = train_test_split(img_total, y_total, test_size=0.2, random_state=22, stratify=y_total)

adam = adam_v2.Adam(learning_rate=0.00001)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3), padding="same"))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding="same"))
#model.add(BatchNormalization(axis=-1))
#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding="same"))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding="same"))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.25))
model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation='softmax'))

#model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=["AUC"])
#model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=[tf.keras.metrics.AUC()])
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['AUC'])

#history = model.fit(X_train, y_train, batch_size=32, validation_split=0.15, epochs=15, verbose=1, class_weight={0:1,1:5})
history = model.fit(X_train, y_train, batch_size=16, validation_split=0.2, epochs=40, verbose=1, class_weight={0: 1, 1: 5})

model.save('CNN.h5')
