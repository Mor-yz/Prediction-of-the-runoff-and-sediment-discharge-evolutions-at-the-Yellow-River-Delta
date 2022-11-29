import numpy as np
from PyEMD import EMD, Visualisation
import xlrd
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn import svm
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import explained_variance_score
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
import math
from matplotlib.pylab import mpl
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K
from keras.layers import LeakyReLU
from keras.callbacks import LearningRateScheduler
from keras.callbacks import EarlyStopping
from tensorflow.keras import Input, Model, Sequential

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# read excel
data = pd.read_excel(r'runoff_res.xlsx', sheet_name='Sheet1')  # change the name of file to use sediment or runoff
data = np.array(data)
data = data.T
y1_res = data.reshape(-1)

t = np.arange(96)
# EMD
emd = EMD()
emd.emd(y1_res)
imfs, res = emd.get_imfs_and_residue()

# figure
vis = Visualisation()
vis.plot_imfs(imfs=imfs, residue=res, t=t, include_residue=True)


vis.plot_instant_freq(t, imfs=imfs)
vis.show()

y1 = imfs[0]
y2 = imfs[1]
y3 = imfs[2]
y4 = imfs[3]
y5 = res

series1 = pd.Series(y1)
series2 = pd.Series(y2)
series3 = pd.Series(y3)
series4 = pd.Series(y4)
series5 = pd.Series(y5)

# 滞后扩充数据
dataframe1 = pd.DataFrame()
num_hour = 6
for i in range(num_hour, 0, -1):
    dataframe1['t-' + str(i)] = series1.shift(i)  # change series1 to IMF number we want. e.g. series1 means IMF1
dataframe1['t'] = series1.values  # change series1 to IMF number we want. e.g. series1 means IMF1
dataframe3 = dataframe1.dropna()
dataframe3.index = range(len(dataframe3))

# spilt
pot = int(len(dataframe3) * 0.75)-1
train = dataframe3[:pot]
test = dataframe3[pot:]
scaler = MinMaxScaler(feature_range=(0, 1)).fit(train)
# scaler = preprocessing.StandardScaler().fit(train)
train_norm = pd.DataFrame(scaler.fit_transform(train))
test_norm = pd.DataFrame(scaler.transform(test))

X_train = train_norm.iloc[:, 1:]
X_test = test_norm.iloc[:, 1:]
Y_train = train_norm.iloc[:, :1]
Y_test = test_norm.iloc[:, :1]

# transfer to 3D
source_x_train = X_train
source_x_test = X_test
X_train = X_train.values.reshape([X_train.shape[0], 3, 2])
X_test = X_test.values.reshape([X_test.shape[0], 3, 2])
Y_train = Y_train.values
Y_test = Y_test.values


# learning rate
def scheduler(epoch):
    if epoch % 50 == 0 and epoch != 0:
        lr = K.get_value(gru.optimizer.lr)
        if lr > 1e-5:
            K.set_value(gru.optimizer.lr, lr * 0.1)
            print("lr changed to {}".format(lr * 0.1))
    return K.get_value(gru.optimizer.lr)


reduce_lr = LearningRateScheduler(scheduler)
early_stopping = EarlyStopping(monitor='loss',
                               patience=20,
                               min_delta=1e-5,
                               mode='auto',
                               restore_best_weights=False,
                               verbose=2)

# GRU
# feature
input_dim = X_train.shape[2]
# time step
time_steps = X_train.shape[1]
batch_size = 1

gru = Sequential()
input_layer = Input(batch_shape=(batch_size, time_steps, input_dim))
gru.add(input_layer)
gru.add(tf.keras.layers.GRU(64))
gru.add(tf.keras.layers.Dense(32))
gru.add(tf.keras.layers.LeakyReLU(alpha=0.3))
gru.add(tf.keras.layers.Dense(16))
gru.add(tf.keras.layers.LeakyReLU(alpha=0.3))
gru.add(tf.keras.layers.Dense(1))
gru.add(tf.keras.layers.LeakyReLU(alpha=0.3))


nadam = tf.keras.optimizers.Nadam(lr=1e-3)
gru.compile(loss='mse', optimizer=nadam, metrics=['mae'])
gru.summary()

boot = np.arange(0, 72, 1, int)
Y_test_bo = []
# train
# bootstrap
for i in range(50):
    boot_train = np.random.choice(boot, 18, replace=True)
    X_train_bo = []
    Y_train_bo = []
    for j in range(18):
        X_train_bo = list(X_train_bo)
        Y_train_bo = list(Y_train_bo)
        X_train_bo.append(list(X_train[boot_train[j]]))
        Y_train_bo.append(list(Y_train[boot_train[j]]))
        X_train_bo = np.array(X_train_bo)
        Y_train_bo = np.array(Y_train_bo)
    history = gru.fit(X_train, Y_train, validation_split=0.1, epochs=100, batch_size=1, callbacks=[reduce_lr])

    # prediction
    predict = gru.predict(X_test, batch_size=1)
    real_predict = scaler.inverse_transform(np.concatenate((source_x_test, predict), axis=1))
    real_y = scaler.inverse_transform(np.concatenate((source_x_test, Y_test), axis=1))
    # real_y = scaler.inverse_transform(Y_test)
    real_predict = real_predict[:, -1]
    real_y = real_y[:, -1]
    Y_test_bo.append(real_predict)
real_predict = np.mean(Y_test_bo, axis=0)

# figure
plt.figure(figsize=(15, 6))
bwith = 0.75
ax = plt.gca()
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
plt.plot(real_predict, label='real_predict')
plt.plot(real_y, label='real_y')
plt.plot(real_y * (1 + 0.15), label='15%up', linestyle='--', color='green')
plt.plot(real_y * (1 - 0.15), label='15%down', linestyle='--', color='green')
plt.legend()
plt.show()

