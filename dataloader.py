import os
import numpy as np


def data_loader():
    """
        load the data from the local csv files.
        My csv files' path is './csv', please adapt to your own path

        Returns:
            data_train (array): size(1000, 11), data for training
            label_train (array): size(1000,), label of the training dataset
            data_val (array): size(662, 11), data for testing
            label_val (array): size(662,), label of the testing dataset

            Note: the whole dataset has 1662 samples, we randomly get 1000 samples without repeating for training, and
            get the rest 662 samples for testing.
    """
    # load csv
    data_all = np.empty(shape=[1, 14])
    for info in os.listdir('./csv'):
        domain = os.path.abspath(r'./csv')
        info = os.path.join(domain, info)
        data = np.loadtxt(info, delimiter=",", skiprows=1)
        data_all = np.vstack((data_all, data))

    data_all = np.delete(data_all, 0, 0)
    data_all = np.delete(data_all, np.s_[0:2], axis=1)

    # preprocessing
    label = data_all[:, 0]
    feature = data_all[:, 1:]
    # feature = normalize(feature, axis=0, norm='max')
    feature[:, [3, 4, 5, 10]] = (feature[:, [3, 4, 5, 10]] - feature[:, [3, 4, 5, 10]].min(axis=0)) / \
                    (feature[:, [3, 4, 5, 10]].max(axis=0) - feature[:, [3, 4, 5, 10]].min(axis=0))
    # feature_norm = (feature - feature.min(axis=0)) / (feature.max(axis=0) - feature.min(axis=0))
    arr = np.array(range(0, 1662))
    arr = np.random.permutation(arr)
    feature = feature[arr]
    label = label[arr].astype(int)

    # load training and testing data
    data_train = feature[0:1000]
    label_train = label[0:1000]
    data_test = feature[1000:]
    label_test = label[1000:]

    data_val = data_test
    label_val = label_test

    return data_train, label_train, data_val, label_val


def data_process(data_train, label_train, data_val, label_val, feature_list, feature_type_mode, sublist_map):
    """
        further process the data to adapt the stepwise

        Args:
            data_train (array): size(1000, 11), data for training
            label_train (array): size(662,), label of data_train
            data_val (array): size(662, 11), same as the data for training
            label_val (array): size(662,), label of data_val
            feature_list (list): the list of the feature index
            feature_type_mode (str): two modes: list, sublist
            sublist_map (list): indicate the index and the value of the major feature

        Returns:
            data_train (array): size(1000, 11), data for training
            label_train (array): size(1000,), label of the training dataset
            data_val (array): size(662, 11), data for testing
            label_val (array): size(662,), label of the testing dataset

            Note: the whole dataset has 1662 samples, we randomly get 1000 samples without repeating for training, and
            get the rest 662 samples for testing.
    """
    if feature_type_mode == 'list':  # if the list is major feature list, we just need to get the feature_list data
        data_train_processed = data_train[:, feature_list]
        label_train_processed = label_train[:]
        data_val_processed = data_val[:, feature_list]
        label_val_processed = label_val[:]
    else:
        if isinstance(feature_list, int):  # if the list is a minor feature list, we need to first know if the feature_list is an integer
            index, value = sublist_map[feature_list]
            if value == -1:  # if the one wanted feature is a major feature, we just need to get the feature_list data
                data_train_processed = data_train[:, index]
                label_train_processed = label_train[:]
                data_val_processed = data_val[:, index]
                label_val_processed = label_val[:]
            else:  # if the one wanted feature is a minor feature, we need to get the feature_list data with the minor feature value
                data_squ = (data_train[:, index] == 0) | (data_train[:, index] == value)
                data_train_processed = data_train[data_squ, index]
                label_train_processed = label_train[data_squ]
                test_squ = (data_val[:, index] == 0) | (data_val[:, index] == value)
                data_val_processed = data_val[test_squ, index]
                label_val_processed = label_val[test_squ, index]
        else:  # when the list is a minor feature list, and the feature_list has several minor features, the process is much complex
            # we first need to know if the minor feature list has two or more minor features that are belong to the same major features
            for i in range(len(feature_list) - 1):
                index, value = sublist_map[feature_list[i]]
                index_later, value_later = sublist_map[feature_list[i + 1]]
                if index == index_later:
                    return 0, 0, 0, 0
            # if the minor features don't conflict in the same major features, we extract the data of the minor feature list
            index, value = [], []
            for i in range(len(feature_list)):
                index_i, value_i = sublist_map[feature_list[i]]
                index.append(index_i)
                value.append(value_i)
            data_train_processed = data_train[:, index]
            label_train_processed = label_train[:]
            data_val_processed = data_val[:, index]
            label_val_processed = label_val[:]
            for i in range(len(feature_list)):
                data_squ = (data_train_processed[:, i] == 0) | (data_train_processed[:, i] == value[i])
                data_train_processed = data_train_processed[data_squ]
                label_train_processed = label_train_processed[data_squ]
                test_squ = (data_val[:, index] == 0) | (data_val[:, index] == value)
                data_val_processed = data_val_processed[test_squ]
                label_val_processed = label_val_processed[test_squ]
    return data_train_processed, label_train_processed, data_val_processed, label_val_processed
