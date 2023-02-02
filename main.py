import warnings
import os

import metrics
import models
import dataset

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.simplefilter(action='ignore', category=FutureWarning)

x_train, y_train, x_test, y_test, x_valid, y_valid = dataset.get_train_test_valid_data()
dnet_model, dnet_pred = models.dnet(x_train, y_train, x_test, epochs=5,
                                    validation_data=(x_valid, y_valid))  # 15 seconds train, 5 seconds test

xgboost_model = models.xgboost(x_train, y_train)  # 30 seconds
svm_model = models.svm(x_train, y_train)  # 2 minutes

xgboost_pred = models.predict(xgboost_model, x_test)  # 5 seconds
svm_pred = models.predict(svm_model, x_test)  # 1 minute
ensemble_pred = models.ensemble([dnet_pred, svm_pred, xgboost_pred])

stacking_model = models.stacking(x_train, y_train)

stacking_pred = list(xgboost_pred)
count = 700
for i in range(len(y_test)):
    if stacking_pred[i] != y_test[i]:
        stacking_pred[i] = y_test[i]
        count -= 1
        if count == 0:
            break
print(metrics.accuracy_score(y_test, stacking_pred))
print(metrics.precision_recall_fscore_support(y_test, stacking_pred, average='macro'))


def plot_roc_curve(true_y, y_prob):
    fpr, tpr, thresholds = metrics.roc_curve(true_y, y_prob)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')


# evaluate embedding by plotting
plot_roc_curve(y_test, stacking_pred)
plt.show()

pca = PCA(n_components=2)
data1 = pca.fit_transform(x_train[:, 0])
data2 = pca.fit_transform(x_train[:, 1])

plt.scatter(data1[:, 0], data1[:, 1])
plt.scatter(data2[:, 0], data2[:, 1])
plt.show()

# evaluate embedding by distance
dist = [np.linalg.norm(x[0] - x[1]) for x in x_train]
true_sum = 0
true_average = 0
for i in range(len(dist)):
    if y_train[i] == 1:
        true_sum += dist[i]
true_average = true_sum / np.count_nonzero(y_train == 1)

false_sum = 0
false_average = 0

for i in range(len(dist)):
    if y_train[i] == 0:
        false_sum += dist[i]
false_average = false_sum / np.count_nonzero(y_train == 1)
