# encoding: utf-8

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
#import tensorflowjs as tfjs

# use diabetes sample data from sklearn
diabetes = load_diabetes()
# load them to X and Y
X = diabetes.data
Y = diabetes.target

print(X.shape , Y.shape )
#quit()
# create regression model
#def reg_model():
model = Sequential()
model.add(Dense(10, input_dim=10, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
# compile model
model.compile(loss='mean_squared_error', optimizer='adam')

# main
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)
# train
batch_size=10
epoch_num=100
#fit
history=model.fit(x_train,y_train
    ,batch_size=batch_size, nb_epoch=epoch_num, verbose=0 )
#モデルと重みを保存
json_string=model.to_json()
open('params.json',"w").write(json_string)
tfjs_target_dir = "./dist"

# モデルの評価
y_pred = model.predict(x_test)
print(y_pred.shape )
# show its root mean square error
mse = mean_squared_error(y_test, y_pred)
print("KERAS REG RMSE : %.2f" % (mse ** 0.5))
print("#end_acc")
