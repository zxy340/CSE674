import numpy as np
import keras
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score
import itertools
from keras.models import Sequential, Model
from keras.layers import Dense
from dataloader import data_loader, data_process

Feature_list = ['gender', 'race', 'intvrage', 'weight', 'wrdimmrc', 'wrddlyrc', 'clkdraw', 'clkimgcl', 'height']

# .................custom parameters.................................
start = 9                                # the minimum number of combination features
end = 9                                  # the maximum number of combination features
# ...................................................................

# load data from csv
data_train, label_train, data_val, label_val = data_loader(root_path='../')

# ............Computing MLP accuracy of each kind of feature combination..................
results = []
for k in range(start, end + 1):
    # network size
    input_dim = k
    hidden_dim = 128

    print('current calculated combination number is {}'.format(k))
    features = list(itertools.combinations(range(len(Feature_list)), k))
    for index_list in range(len(features)):
        data_train_processed, label_train_processed, data_val_processed, label_val_processed = \
            data_process(data_train, label_train, data_val, label_val, features[index_list])
        if len(data_train_processed) == 0:
            continue
        print('The shape of the data_train, label_train, data_val, label_val is {}, {}, {}, {}, respectively'.format(
            np.shape(data_train_processed), np.shape(label_train_processed), np.shape(data_val_processed), np.shape(label_val_processed)
        ))

        net = Sequential([
          Dense(hidden_dim, activation='relu', use_bias=False, input_dim=input_dim),  # hidden layer
          Dense(2, activation='softmax', use_bias=False, name='Dense_1'),
          Dense(2, activation='softmax', use_bias=False, name='Dense_2'),  # output layer
        ])

        # loss is cross entropy
        net.compile(
          loss='binary_crossentropy', optimizer='adam', metrics=['AUC']
        )

        net.fit(
          data_train_processed,
          label_train_processed,
          epochs=20,
          batch_size=32,
          class_weight={0: 1, 1: 5},
          validation_data=(data_val_processed, label_val_processed),
          verbose=0
        )

        # get the feature combination
        feature_name = ''
        if isinstance(features[index_list], int):
            feature_name = str(features[index_list])
        else:
            for i in range(len(features[index_list])):
                feature_name = feature_name + str(features[index_list][i]) + ' '

        # model accuracy calculation
        y_pred = np.argmax(net.predict(data_val_processed), axis=1)
        y_val = np.argmax(label_val_processed, axis=1)
        results.append({'feature': feature_name, 'AUC': roc_auc_score(y_val, y_pred), 'number': len(data_train_processed)})
# ........................................................................
print(results)
np.save('result_C99.npy', results)