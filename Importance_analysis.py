import numpy as np
import keras
import matplotlib.pyplot as plt
from dataloader import data_loader
from Analysis_method import shuffle_list_label_final, zero_list_label_final, shuffle_list_data_final, \
    zero_list_data_final, zero_sublist_data_final, shuffle_list_data_layer, zero_list_data_layer, \
    zero_sublist_data_layer

Feature_list = ['gender', 'race', 'intvrage', 'weight', 'wrdimmrc', 'wrddlyrc', 'clkdraw', 'clkimgcl', 'height']
Feature_sublist = ['gender1', 'gender2', 'race1', 'race2', 'race3',
                   'intvrage1', 'intvrage2', 'intvrage3', 'intvrage4', 'intvrage5', 'intvrage6',
                   'weight',
                   'wrdimmrc1', 'wrdimmrc2', 'wrdimmrc3', 'wrdimmrc4', 'wrdimmrc5', 'wrdimmrc6',
                   'wrdimmrc7', 'wrdimmrc8', 'wrdimmrc9', 'wrdimmrc10',
                   'wrddlyrc1', 'wrddlyrc2', 'wrddlyrc3', 'wrddlyrc4', 'wrddlyrc5', 'wrddlyrc6',
                   'wrddlyrc7', 'wrddlyrc8', 'wrddlyrc9', 'wrddlyrc10',
                   'clkdraw1', 'clkdraw2', 'clkdraw3', 'clkdraw4', 'clkdraw5',
                   'clkimgcl1', 'clkimgcl2', 'clkimgcl3', 'clkimgcl4', 'height']
sublist_map = [[0, 1], [0, 2], [1, 1], [1, 2], [1, 3],
               [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6],
               [3, -1],
               [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 6],
               [4, 7], [4, 8], [4, 9], [4, 10],
               [5, 1], [5, 2], [5, 3], [5, 4], [5, 5], [5, 6],
               [5, 7], [5, 8], [5, 9], [5, 10],
               [6, 1], [6, 2], [6, 3], [6, 4], [6, 5],
               [7, 1], [7, 2], [7, 3], [7, 4], [8, -1]]

# .................custom parameters.................................
data_processing_mode = 'zero'            # two modes: shuffle, zero
feature_type_mode = 'list'               # two modes: list, sublist
importance_calculation_mode = 'data'     # two modes: label, data
is_layer_mode = 'final'                  # two modes: final, layer
layer_index = 1  # the index of the layer to calculate the importance
# ....................................................................

# some fixed parameters
feature_type = {'list': Feature_list, 'sublist': Feature_sublist}
analysis_mode = data_processing_mode + '_' + feature_type_mode + '_' + importance_calculation_mode + '_' + is_layer_mode

# load data from csv
data_train, label_train, data_val, label_val = data_loader()

# load MLP model
model = keras.models.load_model('FNN.h5')

# ............Computing MLP feature importance........................
print('Computing MLP feature importance...')
if analysis_mode == 'shuffle_list_label_final':
    results = shuffle_list_label_final(feature_type[feature_type_mode], data_val, label_val, model)
elif analysis_mode == 'zero_list_label_final':
    results = zero_list_label_final(feature_type[feature_type_mode], data_val, label_val, model)
elif analysis_mode == 'shuffle_list_data_final':
    results = shuffle_list_data_final(feature_type[feature_type_mode], data_val, model)
elif analysis_mode == 'zero_list_data_final':
    results = zero_list_data_final(feature_type[feature_type_mode], data_val, model)
elif analysis_mode == 'zero_sublist_data_final':
    results = zero_sublist_data_final(feature_type[feature_type_mode], sublist_map, data_val, model)
elif analysis_mode == 'shuffle_list_data_layer':
    results = shuffle_list_data_layer(feature_type[feature_type_mode], data_val, model, layer_index)
elif analysis_mode == 'zero_list_data_layer':
    results = zero_list_data_layer(feature_type[feature_type_mode], data_val, model, layer_index)
elif analysis_mode == 'zero_sublist_data_layer':
    results = zero_sublist_data_layer(feature_type[feature_type_mode], sublist_map, data_val, model, layer_index)
# ..............................................................

# ...........plot the importance figure.........................
feature_name = []
importance = []
data_number = []
index = np.arange(len(feature_type[feature_type_mode]))
for i in range(len(feature_type[feature_type_mode])):
    feature_name.append(results[i]['feature'])
    importance.append(results[i]['mae'])
    data_number.append(results[i]['number'])
plt.figure(figsize=(45, 5))
plt.bar(x=index, height=importance)
plt.xticks(index, feature_name)
for x, y, z in zip(index, importance, data_number):
    plt.text(x, y, '%d' % z, ha='center')
plt.show()
# ..............................................................