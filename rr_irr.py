import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn import svm
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error  # 平方绝对误差
import math
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import LinearRegression


def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100


def ns(x, y):
    output_errors_up = (x - y) ** 2
    aver = np.average(x)
    output_errors_down = []
    a = list(range(len(x)))
    for i in range(len(x)):
        output_errors_down.append((x[i] - aver) ** 2)
    # for i in range(len(x)):
    #     output_errors_down[i] = a[i] ** 2
    out = 1 - sum(output_errors_up) / sum(output_errors_down)
    return out


def r(x, y):
    output_errors_up = np.average((x - np.average(x)) * (y - np.average(y)))
    output_errors_down = math.sqrt(np.average((x - np.average(x)) ** 2)) * math.sqrt(
        np.average((y - np.average(y)) ** 2))
    return output_errors_up / output_errors_down


# 读文件
data = pd.read_excel(r'data/original data/log.xlsx')
datairri = pd.read_excel(r'data/process data/kernel.xlsx', sheet_name='spherical')  # choose kernel
irrigation = np.array(datairri)
irrigation = irrigation[:, 2:]
data = np.array(data)
data1 = data[:, 1]  # sediment
data2 = data[:, 2]  # runoff

effect = '000110000101001000100000010010010100001010100100010010001000001100100010000000001110010000000010100100011001101000001010000000000000000100010010001100000010100000000001000011100000001000001100000000000100010000011000110100000001000000100000000010010010011110000000010010001000100000000001000000000000100001010100001001000000000010000000000000000101000000000100001000000000000100000000000010000000000100110101000010000000000000000000010000000000000001011000001111001011000100000000000001000101000'
effect = list(effect)

x = []
for i in range(96):
    v = irrigation[i]
    x.append(v)
x = np.array(x)  # sac variable
x_choosen = []
for i in range(96):
    v = []
    for j in range(495):
        if effect[j] == '1':
            # if (j+1) in choose:
            v.append(x[i][j])
    x_choosen.append(v)
x_choosen = np.array(x_choosen)

x = x_choosen  # use feature selected by QBSO or not

y1 = []
for i in range(96):
    y1.append(data1[i + 60])
y1 = np.array(y1)  # sediment

y2 = []
for i in range(96):
    y2.append(data2[i + 60])
y2 = np.array(y2)  # runoff

# spilt
x_train = x[:72]
x_test = x[72:]
y1_train = y1[:72]
y1_test = y1[72:]
y2_train = y2[:72]
y2_test = y2[72:]

# RR
model = RidgeCV(alphas=[0.1, 1.0, 10.0])
model.fit(x_train, y1_train)
y1_hat = model.predict(x)

y1 = y1.reshape(-1, 1)
y1_hat = y1_hat.reshape(-1, 1)
print("RMSE:", math.sqrt(mean_squared_error(y1[72:], y1_hat[72:])))
print("MAPE = ", mape(y1[72:], y1_hat[72:]))
print("R = ", r(y1[72:], y1_hat[72:]))
print("NSCE = ", ns(y1[72:], y1_hat[72:]))

t = np.arange(96)  # year 2011-2018
plt.figure(facecolor='w')
plt.plot(t[:72], y1_hat[0:72], 'r-', linewidth=2, label='train')
plt.plot(t[72:], y1_hat[72:], 'b-', linewidth=2, label='test')
plt.plot(t, y1, 'g-', label='true')
plt.legend(loc='upper right')
plt.title('sediment', fontsize=16)
plt.xlabel('t')
plt.ylabel('sediment')
plt.grid(True)
plt.show()

# RR
model = RidgeCV(alphas=[0.1, 1.0, 10.0])
model.fit(x_train, y2_train)
y2_hat = model.predict(x)

y2 = y2.reshape(-1, 1)
y2_hat = y2_hat.reshape(-1, 1)
print("RMSE:", math.sqrt(mean_squared_error(y2[72:], y2_hat[72:])))
print("MAPE = ", mape(y2[72:], y2_hat[72:]))
print("R = ", r(y2[72:], y2_hat[72:]))
print("NSCE = ", ns(y2[72:], y2_hat[72:]))

t = np.arange(96)  # year 2011-2018
plt.figure(facecolor='w')
plt.plot(t[:72], y2_hat[0:72], 'r-', linewidth=2, label='train')
plt.plot(t[72:], y2_hat[72:], 'b-', linewidth=2, label='test')
plt.plot(t, y2, 'g-', label='true')
plt.legend(loc='upper right')
plt.title('runoff', fontsize=16)
plt.xlabel('t')
plt.ylabel('runoff')
plt.grid(True)
plt.show()

# model = svm.SVR(kernel='rbf', epsilon=0.1)
# c_can = np.logspace(-2, 2, 10)
# gamma_can = np.logspace(-2, 2, 10)
# svr = GridSearchCV(model, param_grid={'C': c_can, 'gamma': gamma_can}, cv=5)
# svr.fit(x_train, y1_train)
# y1_hat = svr.predict(x)
#
# # # figure
# # t = np.arange(96)  # year 2011-2018
# # plt.figure(facecolor='w')
# # plt.plot(t[:72], y1_hat[0:72], 'r-', linewidth=2, label='train')
# # plt.plot(t[72:], y1_hat[72:], 'b-', linewidth=2, label='test')
# # plt.plot(t, y1, 'g-', label='true')
# # plt.legend(loc='upper right')
# # plt.title('runoff', fontsize=16)
# # plt.xlabel('t')
# # plt.ylabel('runoff')
# # plt.grid(True)
# # plt.show()
# y1 = y1.reshape(-1, 1)
# y1_hat = y1_hat.reshape(-1, 1)
# print("RMSE:", math.sqrt(mean_squared_error(y1[72:], y1_hat[72:])))
# print("MAPE = ", mape(y1[72:], y1_hat[72:]))
# print("R = ", math.sqrt(metrics.r2_score(y1[72:], y1_hat[72:])))
# print("NSCE = ", ns(y1[72:], y1_hat[72:]))
#
# #
# model = svm.SVR(kernel='rbf', epsilon=0.1)
# c_can = np.logspace(-2, 2, 10)
# gamma_can = np.logspace(-2, 2, 10)
# svr = GridSearchCV(model, param_grid={'C': c_can, 'gamma': gamma_can}, cv=5)
# svr.fit(x_train, y2_train)
# y2_hat = svr.predict(x)
# #作图
# t = np.arange(96)  # year 2011-2018
# plt.figure(facecolor='w')
# plt.plot(t[:72], y1_hat[0:72], 'r-', linewidth=2, label='train')
# plt.plot(t[72:], y1_hat[72:], 'b-', linewidth=2, label='test')
# plt.plot(t, y1, 'g-', label='true')
# plt.legend(loc='upper right')
# plt.title('runoff', fontsize=16)
# plt.xlabel('t')
# plt.ylabel('runoff')
# plt.grid(True)
# plt.show()
# y2 = y2.reshape(-1, 1)
# y2_hat = y2_hat.reshape(-1, 1)
# print("RMSE:", math.sqrt(mean_squared_error(y2[72:], y2_hat[72:])))
# print("MAPE = ", mape(y2[72:], y2_hat[72:]))
# print("R = ", r(y2[72:], y2_hat[72:]))
# print("NSCE = ", ns(y2[72:], y2_hat[72:]))


# # MLR
# model = LinearRegression()
# model.fit(x_train, y1_train)
# y1_hat = model.predict(x)
#
# print("RMSE:", math.sqrt(mean_squared_error(y1[72:], y1_hat[72:])))
# print("MAPE = ", mape(y1[72:], y1_hat[72:]))
# print("R = ", r(y1[72:], y1_hat[72:]))
# print("NSCE = ", ns(y1[72:], y1_hat[72:]))
#
# t = np.arange(96)  # year 2011-2018
# plt.figure(facecolor='w')
# plt.plot(t[:72], y1_train, 'r-', linewidth=2, label='train')
# plt.plot(t[72:], y1_hat[72:], 'b-', linewidth=2, label='test')
# plt.plot(t, y1, 'g-', label='true')
# plt.legend(loc='upper right')
# plt.title('sediment', fontsize=16)
# plt.xlabel('t')
# plt.ylabel('sediment')
# plt.grid(True)
# plt.show()
#
# # MLR
# model = LinearRegression()
# model.fit(x_train, y2_train)
# y2_hat = model.predict(x)
#
# print("RMSE:", math.sqrt(mean_squared_error(y2[72:], y2_hat[72:])))
# print("MAPE = ", mape(y2[72:], y2_hat[72:]))
# print("R = ", r(y2[72:], y2_hat[72:]))
# print("NSCE = ", ns(y2[72:], y2_hat[72:]))
#
#
# t = np.arange(96)  # year 2011-2018
# plt.figure(facecolor='w')
# plt.plot(t[:72], y2_hat[0:72], 'r-', linewidth=2, label='train')
# plt.plot(t[72:], y2_hat[72:], 'b-', linewidth=2, label='test')
# plt.plot(t, y2, 'g-', label='true')
# plt.legend(loc='upper right')
# plt.title('sediment', fontsize=16)
# plt.xlabel('t')
# plt.ylabel('sediment')
# plt.grid(True)
# plt.show()
