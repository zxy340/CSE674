import numpy as np
import matplotlib.pyplot as plt

results = []
result_C99 = np.load('result_C99.npy', allow_pickle=True)
result_C97 = np.load('result_C97.npy', allow_pickle=True)
result_C94 = np.load('result_C94.npy', allow_pickle=True)
result_C97 = sorted(result_C97, key=lambda i: i['AUC'], reverse=True)
result_C94 = sorted(result_C94, key=lambda i: i['AUC'], reverse=True)
results.append(result_C99[0])
for i in range(5):  # top 5 for C97
    results.append(result_C97[i])
for i in range(5):  # top 5 for C94
    results.append(result_C94[i])

# ...........plot the accuracy of all combination.........................
feature_name, accuracy, data_number = [], [], []
index = np.arange(len(results))
for i in range(len(results)):
    feature_name.append(results[i]['feature'])
    accuracy.append(results[i]['AUC'])
    data_number.append(results[i]['number'])
plt.figure(figsize=(20, 5))
plt.bar(x=index, height=accuracy)
plt.xticks(index, feature_name)
for x, y, z in zip(index, accuracy, data_number):
    plt.text(x, y, '%d' % z, ha='center')
plt.title('Stepwise results')
plt.xlabel('Features')
plt.ylabel('AUC score')
plt.show()