import os
import numpy as np
from tensorflow.keras.preprocessing import image

def data_loader(root_path=''):
    """
        load the data from the local csv files.
        My csv files' path is './csv', please adapt to your own path

        Returns:
            data_train (array): size(1200, 9), data for training
            label_train (array): size(1200,), label of the training dataset
            data_val (array): size(462, 9), data for testing
            label_val (array): size(462,), label of the testing dataset

            Note: the whole dataset has 1662 samples, we randomly get 1200 samples without repeating for training, and
            get the rest 462 samples for testing.
    """
    # load csv
    data_all = np.empty(shape=[1, 14])
    for info in os.listdir(root_path + './csv'):
        # domain = os.path.abspath(r'./csv')
        domain = root_path + './csv/'
        info = os.path.join(domain, info)
        data = np.loadtxt(info, delimiter=",", skiprows=1)
        data_all = np.vstack((data_all, data))

    data_all = np.delete(data_all, 0, 0)
    data_all = np.delete(data_all, np.s_[0, 1, 7, 8], axis=1)

    # preprocessing
    label = data_all[:, 0]
    feature = data_all[:, 1:]
    # feature = normalize(feature, axis=0, norm='max')
    feature[:, [3, 8]] = (feature[:, [3, 8]] - feature[:, [3, 8]].min(axis=0)) / \
                    (feature[:, [3, 8]].max(axis=0) - feature[:, [3, 8]].min(axis=0))
    # feature_norm = (feature - feature.min(axis=0)) / (feature.max(axis=0) - feature.min(axis=0))
    arr = np.array(range(0, 1662))
    arr = np.random.permutation(arr)
    feature = feature[arr]
    label = label[arr].astype(int)

    # load training and testing data
    data_train = feature[0:1200]
    label_train = label[0:1200]
    data_test = feature[1200:]
    label_test = label[1200:]

    data_val = data_test
    label_val = label_test

    return data_train, label_train, data_val, label_val


def data_process(data_train, label_train, data_val, label_val, feature_list):
    """
        further process the data to adapt the stepwise

        Args:
            data_train (array): size(1200, 9), data for training
            label_train (array): size(1200,), label of data_train
            data_val (array): size(462, 9), same as the data for training
            label_val (array): size(462,), label of data_val
            feature_list (list): the list of the feature index

        Returns:
            data_train (array): size(1200, 9), data for training
            label_train (array): size(1200, 2), label of the training dataset
            data_val (array): size(462, 9), data for testing
            label_val (array): size(462, 2), label of the testing dataset

            Note: the whole dataset has 1662 samples, we randomly get 1200 samples without repeating for training, and
            get the rest 462 samples for testing.
    """
    data_train_processed = data_train[:, feature_list]
    label_train_processed = np.eye(2)[label_train.reshape(-1)]
    data_val_processed = data_val[:, feature_list]
    label_val_processed = np.eye(2)[label_val.reshape(-1)]
    return data_train_processed, label_train_processed, data_val_processed, label_val_processed

def data_loader_CNNimage(root_path='', CNN_model=None):
    # load csv
    data_all = np.empty(shape=[1, 14])
    for info in os.listdir(root_path + './csv'):
        domain = root_path + './csv/'
        info = os.path.join(domain, info)
        data = np.loadtxt(info, delimiter=",", skiprows=1)
        data_all = np.vstack((data_all, data))

    data_all = np.delete(data_all, 0, 0)
    allimg = []
    for index in range(len(data_all)):
        img = image.load_img(r'./crop/' + str(int(data_all[index, 1])) + '.jpg',
                             target_size=(224, 224, 3))
        img = image.img_to_array(img)
        img = img / 255
        allimg.append(img)
    img_total = np.array(allimg)
    img_feature = np.argmax(CNN_model.predict(img_total), axis=1)
    img_feature = img_feature[:, np.newaxis]
    data_all = np.concatenate((data_all, img_feature), axis=1)
    data_all = np.delete(data_all, np.s_[0, 1, 7, 8], axis=1)

    # preprocessing
    label = data_all[:, 0]
    feature = data_all[:, 1:]
    # feature = normalize(feature, axis=0, norm='max')
    feature[:, [3, 8]] = (feature[:, [3, 8]] - feature[:, [3, 8]].min(axis=0)) / \
                         (feature[:, [3, 8]].max(axis=0) - feature[:, [3, 8]].min(axis=0))
    # feature_norm = (feature - feature.min(axis=0)) / (feature.max(axis=0) - feature.min(axis=0))
    arr = np.array(range(0, 1662))
    arr = np.random.permutation(arr)
    feature = feature[arr]
    label = label[arr].astype(int)

    # load training and testing data
    data_train = feature[0:1200]
    label_train = label[0:1200]
    data_test = feature[1200:]
    label_test = label[1200:]

    data_val = data_test
    label_val = label_test

    return data_train, label_train, data_val, label_val

def data_loader_meanimage(root_path=''):
    # load csv
    data_all = np.empty(shape=[1, 14])
    for info in os.listdir(root_path + './csv'):
        domain = root_path + './csv/'
        info = os.path.join(domain, info)
        data = np.loadtxt(info, delimiter=",", skiprows=1)
        data_all = np.vstack((data_all, data))

    data_all = np.delete(data_all, 0, 0)
    img_feature = np.zeros((len(data_all), 3))
    for index in range(len(data_all)):
        img = image.load_img(r'./crop/' + str(int(data_all[index, 1])) + '.jpg',
                             target_size=(224, 224, 3))
        img = image.img_to_array(img)
        img = img / 255

        count = 0
        for ii in range(len(img)):
            for jj in range(len(img[0])):
                if np.any(img[ii, jj, :] < 1):
                    count = count + 1
                    img_feature[index, :] = img_feature[index, :] + img[ii, jj, :]
        img_feature[index, :] = img_feature[index, :] / count

    data_all = np.concatenate((data_all, img_feature), axis=1)
    data_all = np.delete(data_all, np.s_[0, 1, 7, 8], axis=1)

    # preprocessing
    label = data_all[:, 0]
    feature = data_all[:, 1:]
    # feature = normalize(feature, axis=0, norm='max')
    feature[:, [3, 8]] = (feature[:, [3, 8]] - feature[:, [3, 8]].min(axis=0)) / \
                         (feature[:, [3, 8]].max(axis=0) - feature[:, [3, 8]].min(axis=0))
    # feature_norm = (feature - feature.min(axis=0)) / (feature.max(axis=0) - feature.min(axis=0))
    arr = np.array(range(0, 1662))
    arr = np.random.permutation(arr)
    feature = feature[arr]
    label = label[arr].astype(int)

    # load training and testing data
    data_train = feature[0:1200]
    label_train = label[0:1200]
    data_test = feature[1200:]
    label_test = label[1200:]

    data_val = data_test
    label_val = label_test

    return data_train, label_train, data_val, label_val