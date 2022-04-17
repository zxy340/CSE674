import numpy as np
from keras.models import Model

"""
    shuffle:
        Example
        --------
        >>> a = np.array([[1, 2], [3, 4]])
        a has the size of (2, 2), while the first 2 is the number of samples, the second 2 is the number of features
        if we want to calculate the importance of the first feature, after shuffle the first feature, we randomly get:
        array([3, 2], [1, 4])
        we compare the output change of these two input data to find the importance to the first feature
    
    zero:
        Example
        --------
        >>> a = np.array([[1, 2], [3, 4]])
        same as the shuffle's example, after shuffle the first feature, we can get:
        array([0, 2], [0, 4])
        we compare the output change of these two input data to find the importance to the first feature
        
    list: indicate we want to calculate the major features importance
    
    sublist: indicate we want to calculate the minor features importance, note there is some difference with the list mode,
    please refer to the below example.
        Example
        --------
        >>> a = np.array([[1, 2], [3, 4], [1, 5]])
        the first feature has three values: 1, 2, 3. As we can see from the example, a has two samples with value 1 and 
        one sample with value 3, we want to calculate the importance of the first feature with value 1, after zero mode,
        we can get:
        array([0, 2], [0, 5])
        together with the raw input without value 3 of the first feature: array([1. 2], [1, 5])
        we compare the output change of these two input data to find the importance to the first feature with value 1
        
    label: this mode indicates how we calculate the output change
        For example, after changing the input to array([0, 2], [0, 5]), we get output_new = np.array([0.7, 0.3], [0.8, 0.2])
        the label of the raw input is np.array([1, 0], [1, 0]), then the importance can be calculated in this way:
        np.mean(np.norm(label - output_new, axis=1))
        
    data: 
        different with label mode, we also input the raw input and get the output_raw to calculate the importance,
        for example, we get the output_raw = np.array([0.9, 0.1], [0.6, 0.4]), then the importance can be calculated in this way:
        np.mean(np.norm(output_raw - output_new, axis=1))
    
    final: indicate we get the importance of the last layer
    
    layer: indicate we get the importance of the selected middle layer
"""
def shuffle_list_label_final(features, data_val, label_val, model):
    """
        Args:
            features (list): the list of the feature names
            data_val (array): size(662, 11), same as the data for testing
            label_val (array): size(662,), label of data_val
            model (dict): the trained MLP model

        Returns:
            results (dict):
                {
                    'feature': name of the feature
                    'mae': the importance of the corresponding feature
                    'number': the samples number of the corresponding feature
                }
    """
    results = []
    for k in range(len(features)):
        temp = data_val[:, k].copy()
        np.random.shuffle(data_val[:, k])
        oof_pred = model.predict(data_val, verbose=0).squeeze()
        mae = np.mean(np.linalg.norm(oof_pred - np.eye(2)[label_val.reshape(-1)], axis=1).squeeze())
        results.append({'feature': features[k], 'mae': mae, 'number': len(data_val[:, k])})
        data_val[:, k] = temp
    return results

def zero_list_label_final(features, data_val, label_val, model):
    """
        Args:
            features (list): the list of the feature names
            data_val (array): size(662, 11), same as the data for testing
            label_val (array): size(662,), label of data_val
            model (dict): the trained MLP model

        Returns:
            results (dict):
                {
                    'feature': name of the feature
                    'mae': the importance of the corresponding feature
                    'number': the samples number of the corresponding feature
                }
    """
    results = []
    for k in range(len(features)):
        temp = data_val[:, k].copy()
        data_val[:, k] = np.zeros(np.shape(data_val[:, k]))
        oof_pred = model.predict(data_val, verbose=0).squeeze()
        mae = np.mean(np.linalg.norm(oof_pred - np.eye(2)[label_val.reshape(-1)], axis=1).squeeze())
        results.append({'feature': features[k], 'mae': mae, 'number': len(data_val[:, k])})
        data_val[:, k] = temp
    return results

def shuffle_list_data_final(features, data_val, model):
    """
        Args:
            features (list): the list of the feature names
            data_val (array): size(662, 11), same as the data for testing
            model (dict): the trained MLP model

        Returns:
            results (dict):
                {
                    'feature': name of the feature
                    'mae': the importance of the corresponding feature
                    'number': the samples number of the corresponding feature
                }
    """
    results = []
    for k in range(len(features)):
        temp = data_val[:, k].copy()
        raw_pred = model.predict(data_val, verbose=0).squeeze()
        np.random.shuffle(data_val[:, k])
        oof_pred = model.predict(data_val, verbose=0).squeeze()
        mae = np.mean(np.linalg.norm(oof_pred - raw_pred, axis=1).squeeze())
        results.append({'feature': features[k], 'mae': mae, 'number': len(data_val[:, k])})
        data_val[:, k] = temp
    return results

def zero_list_data_final(features, data_val, model):
    """
        Args:
            features (list): the list of the feature names
            data_val (array): size(662, 11), same as the data for testing
            model (dict): the trained MLP model

        Returns:
            results (dict):
                {
                    'feature': name of the feature
                    'mae': the importance of the corresponding feature
                    'number': the samples number of the corresponding feature
                }
    """
    results = []
    for k in range(len(features)):
        temp = data_val[:, k].copy()
        raw_pred = model.predict(data_val, verbose=0).squeeze()
        data_val[:, k] = np.zeros(np.shape(data_val[:, k]))
        oof_pred = model.predict(data_val, verbose=0).squeeze()
        mae = np.mean(np.linalg.norm(oof_pred - raw_pred, axis=1).squeeze())
        results.append({'feature': features[k], 'mae': mae, 'number': len(data_val[:, k])})
        data_val[:, k] = temp
    return results

def zero_sublist_data_final(features, sublist_map, data_val, model):
    """
        Args:
            features (list): the list of the feature names
            sublist_map (list): indicate the index and the value of the major feature
            data_val (array): size(662, 11), same as the data for testing
            model (dict): the trained MLP model

        Returns:
            results (dict):
                {
                    'feature': name of the feature
                    'mae': the importance of the corresponding feature
                    'number': the samples number of the corresponding feature
                }
    """
    results = []
    for k in range(len(features)):
        index, value = sublist_map[k]
        if value == -1:
            squ = np.arange(len(data_val))
        else:
            squ = data_val[:, index] == value
        temp = data_val[squ].copy()
        if len(data_val[squ]) == 0:
            results.append({'feature': features[k], 'mae': 0, 'number': 0})
            continue
        raw_pred = model.predict(data_val[squ], verbose=0).squeeze()
        data_val[squ, index] = np.zeros(len(data_val[squ]))
        oof_pred = model.predict(data_val[squ], verbose=0).squeeze()
        if len(data_val[squ]) == 1:
            mae = np.mean(np.linalg.norm(oof_pred - raw_pred, axis=0).squeeze())
        else:
            mae = np.mean(np.linalg.norm(oof_pred - raw_pred, axis=1).squeeze())
        results.append({'feature': features[k], 'mae': mae, 'number': len(data_val[squ])})
        data_val[squ] = temp
    return results

def shuffle_list_data_layer(features, data_val, model, layer_index):
    """
        Args:
            features (list): the list of the feature names
            data_val (array): size(662, 11), same as the data for testing
            model (dict): the trained MLP model
            layer_index (int): indicate the index of the layer for importance calculation

        Returns:
            results (dict):
                {
                    'feature': name of the feature
                    'mae': the importance of the corresponding feature
                    'number': the samples number of the corresponding feature
                }
    """
    layer_model = Model(inputs=model.input, outputs=model.layers[layer_index].output)
    results = []
    for k in range(len(features)):
        temp = data_val[:, k].copy()
        raw_pred = layer_model.predict(data_val, verbose=0).squeeze()
        np.random.shuffle(data_val[:, k])
        oof_pred = layer_model.predict(data_val, verbose=0).squeeze()
        mae = np.mean(np.linalg.norm(oof_pred - raw_pred, axis=1).squeeze())
        results.append({'feature': features[k], 'mae': mae, 'number': len(data_val[:, k])})
        data_val[:, k] = temp
    return results

def zero_list_data_layer(features, data_val, model, layer_index):
    """
        Args:
            features (list): the list of the feature names
            data_val (array): size(662, 11), same as the data for testing
            model (dict): the trained MLP model
            layer_index (int): indicate the index of the layer for importance calculation

        Returns:
            results (dict):
                {
                    'feature': name of the feature
                    'mae': the importance of the corresponding feature
                    'number': the samples number of the corresponding feature
                }
    """
    layer_model = Model(inputs=model.input, outputs=model.layers[layer_index].output)
    results = []
    for k in range(len(features)):
        temp = data_val[:, k].copy()
        raw_pred = layer_model.predict(data_val, verbose=0).squeeze()
        data_val[:, k] = np.zeros(np.shape(data_val[:, k]))
        oof_pred = layer_model.predict(data_val, verbose=0).squeeze()
        mae = np.mean(np.linalg.norm(oof_pred - raw_pred, axis=1).squeeze())
        results.append({'feature': features[k], 'mae': mae, 'number': len(data_val[:, k])})
        data_val[:, k] = temp
    return results

def zero_sublist_data_layer(features, sublist_map, data_val, model, layer_index):
    """
        Args:
            features (list): the list of the feature names
            sublist_map (list): indicate the index and the value of the major feature
            data_val (array): size(662, 11), same as the data for testing
            model (dict): the trained MLP model
            layer_index (int): indicate the index of the layer for importance calculation

        Returns:
            results (dict):
                {
                    'feature': name of the feature
                    'mae': the importance of the corresponding feature
                    'number': the samples number of the corresponding feature
                }
    """
    layer_model = Model(inputs=model.input, outputs=model.layers[layer_index].output)
    results = []
    for k in range(len(features)):
        index, value = sublist_map[k]
        if value == -1:
            squ = np.arange(len(data_val))
        else:
            squ = data_val[:, index] == value
        temp = data_val[squ].copy()
        if len(data_val[squ]) == 0:
            results.append({'feature': features[k], 'mae': 0, 'number': 0})
            continue
        raw_pred = layer_model.predict(data_val[squ], verbose=0).squeeze()
        data_val[squ, index] = np.zeros(len(data_val[squ]))
        oof_pred = layer_model.predict(data_val[squ], verbose=0).squeeze()
        if len(data_val[squ]) == 1:
            mae = np.mean(np.linalg.norm(oof_pred - raw_pred, axis=0).squeeze())
        else:
            mae = np.mean(np.linalg.norm(oof_pred - raw_pred, axis=1).squeeze())
        results.append({'feature': features[k], 'mae': mae, 'number': len(data_val[squ])})
        data_val[squ] = temp
    return results