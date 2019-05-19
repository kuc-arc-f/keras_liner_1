# encoding: utf-8
# https://qiita.com/cvusk/items/33867fbec742bda3f307

import numpy as np
import pandas as pds
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes

# use diabetes sample data from sklearn
diabetes = load_diabetes()

# load them to X and Y
X = diabetes.data
Y = diabetes.target

print(X.shape , Y.shape )
# create regression model
def reg_model():
    model = Sequential()
    model.add(Dense(10, input_dim=10, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))

    # compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# main
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)
estimator = KerasRegressor(build_fn=reg_model, epochs=100, batch_size=10, verbose=0)
estimator.fit(x_train, y_train)
y_pred = estimator.predict(x_test)

# show its root mean square error
mse = mean_squared_error(y_test, y_pred)
print("KERAS REG RMSE : %.2f" % (mse ** 0.5))
