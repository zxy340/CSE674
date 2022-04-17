import numpy as np
import keras
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import itertools
from keras.models import Sequential, Model
from keras.layers import Dense
from dataloader import data_loader, data_process

Feature_list = ['gender', 'race', 'intvrage', 'weight', 'howtallft', 'howtallin',
                'wrdimmrc', 'wrddlyrc', 'clkdraw', 'clkimgcl', 'height']
Feature_sublist = ['gender', 'race', 'intvrage', 'weight', 'howtallft', 'howtallin',
                   'wrdimmrc1', 'wrdimmrc2', 'wrdimmrc3', 'wrdimmrc4', 'wrdimmrc5', 'wrdimmrc6',
                   'wrdimmrc7', 'wrdimmrc8', 'wrdimmrc9', 'wrdimmrc10',
                   'wrddlyrc1', 'wrddlyrc2', 'wrddlyrc3', 'wrddlyrc4', 'wrddlyrc5', 'wrddlyrc6',
                   'wrddlyrc7', 'wrddlyrc8', 'wrddlyrc9', 'wrddlyrc10',
                   'clkdraw1', 'clkdraw2', 'clkdraw3', 'clkdraw4', 'clkdraw5',
                   'clkimgcl', 'height']
sublist_map = [[0, -1], [1, -1], [2, -1], [3, -1], [4, -1], [5, -1],
               [6, 1], [6, 2], [6, 3], [6, 4], [6, 5], [6, 6],
               [6, 7], [6, 8], [6, 9], [6, 10],
               [7, 1], [7, 2], [7, 3], [7, 4], [7, 5], [7, 6],
               [7, 7], [7, 8], [7, 9], [7, 10],
               [8, 1], [8, 2], [8, 3], [8, 4], [8, 5],
               [9, -1], [10, -1]]

# .................custom parameters.................................
feature_type_mode = 'list'               # two modes: list, sublist
start = 10                               # the minimum number of combination features
end = 11                                 # the maximum number of combination features
# ...................................................................

# some fixed parameters
feature_type = {'list': Feature_list, 'sublist': Feature_sublist}

# load data from csv
data_train, label_train, data_val, label_val = data_loader()

#network size
input_dim = 11
hidden_dim = 128

# ............Computing MLP accuracy of each kind of feature combination..................
results = []
for k in range(start, end + 1):
    print('current calculated combination number is {}'.format(k))
    features = list(itertools.combinations(range(len(feature_type[feature_type_mode])), k))
    for index_list in range(len(features)):
        data_train_processed, label_train_processed, data_val_processed, label_val_processed = \
            data_process(data_train, label_train, data_val, label_val, features[index_list], feature_type_mode, sublist_map)
        if isinstance(data_train_processed, int):
            continue

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
          validation_data=(data_val, label_val),
          verbose=0
        )

        # get the feature combination
        feature_name = ''
        if isinstance(features[index_list], int):
            feature_name = str(features[index_list])
        else:
            for i in range(len(features[index_list])):
                feature_name = feature_name + str(features[index_list][i]) + ' '
        print(feature_name)

        # model accuracy calculation
        y_pred = np.argmax(net.predict(data_val), axis=1)
        results.append({'feature': feature_name, 'acc': accuracy_score(label_val, y_pred), 'number': len(data_train_processed)})
# ........................................................................

# ...........plot the accuracy of all combination.........................
feature_name = []
accuracy = []
data_number = []
index = np.arange(len(feature_type[feature_type_mode]))
for i in range(len(feature_type[feature_type_mode])):
    feature_name.append(results[i]['feature'])
    accuracy.append(results[i]['acc'])
    data_number.append(results[i]['number'])
plt.figure(figsize=(45, 5))
plt.bar(x=index, height=accuracy)
plt.xticks(index, feature_name)
for x, y, z in zip(index, accuracy, data_number):
    plt.text(x, y, '%d' % z, ha='center')
plt.show()