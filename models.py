from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import numpy as np
from sklearn.svm import SVC
import xgboost as xgb
import pandas as pd
from scikeras.wrappers import KerasClassifier


def get_dnn():
    return keras.Sequential([
        keras.layers.Dense(80, activation='relu'),
        keras.layers.Dense(40, activation='relu'),
        keras.layers.Dense(20, activation='relu'),
        keras.layers.Dense(10, activation='relu'),
        keras.layers.Dense(4, activation='relu'),
        keras.layers.Dense(2, activation='sigmoid')
    ])


def dnet(x_train, y_train, x_test, epochs=5, validation_data=None):
    # scaling data between 0 and 1 for better accuracy
    x_train = x_train / np.max(x_train)
    x_test = x_test / np.max(x_test)

    # building the neural network
    model = get_dnn()

    # compiling the model with appropriate parameters
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']
                  )

    # fitting the model on the training data and defining epochs value
    print(model.fit(x_train, y_train, validation_data=validation_data, epochs=epochs))

    # using model to predict the test data, so we can show a confusion matrix on the training data.
    y_predicted = model.predict(x_test)
    y_predicted_labels = [np.argmax(x) for x in y_predicted]

    return model, y_predicted_labels


def svm(x_train, y_train):
    classifier = SVC(C=0.1, gamma=30, kernel='rbf')
    classifier.fit(x_train, y_train)
    return classifier


def xgboost(x_train, y_train):
    xg_reg = xgb.XGBClassifier(objective='binary:logistic', colsample_bytree=0.3, learning_rate=0.1,
                               alpha=10, n_estimators=10)
    xg_reg.fit(x_train, y_train)
    return xg_reg


def predict(model, x_test):
    return model.predict(x_test)


def ensemble(predictions):
    votes = np.stack((predictions[0], predictions[1], predictions[2]), axis=1)
    votes = pd.DataFrame(data=votes)
    ensemble_pred = votes.mode(axis='columns').to_numpy()
    return ensemble_pred


def stacking(x_train, y_train):
    dnn = get_dnn()
    dnn.build(input_shape=(len(x_train), len(x_train[0])))
    dnn.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
                )
    model = KerasClassifier(model=dnn, epochs=5, batch_size=10, verbose=0)
    estimators = [
        ('dnn', model),
        ('xgb', xgb.XGBClassifier(objective='binary:logistic', colsample_bytree=0.3, learning_rate=0.1,
                                  alpha=10, n_estimators=10)),
        ('svr', make_pipeline(StandardScaler(), SVC(C=0.1, gamma=30, kernel='rbf')))
    ]
    clf = StackingClassifier(estimators=estimators, final_estimator=SVC(C=0.1, gamma=30, kernel='rbf'))
    clf.fit(x_train, y_train)
    return clf
