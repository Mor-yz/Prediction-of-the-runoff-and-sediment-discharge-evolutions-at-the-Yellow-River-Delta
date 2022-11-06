import numpy as np
import xlrd
import pandas as pd
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import os
import re
import xlwt


def euclidean(lat, lon, target):
    k = []
    for j in range(len(lat)):
        k.append(((lat[j] - target[0]) ** 2 + (lon[j] - target[1]) ** 2) ** 0.5)
    return k


def cc(variable):
    # variable = variable[:96]
    pj = []
    for j in range(variable.shape[1] - 1):
        y = 0
        for i in range(96):  # time
            y += variable[i][j + 1]
        y_hat = y / 96
        pj.append(y_hat)
    z = 0
    for i in range(96):  # time
        z += variable[i][900]
    z_hat = z / 96
    m = []
    for j in range(variable.shape[1] - 1):
        fenzi = 0
        fenmu1 = 0
        fenmu2 = 0
        for i in range(96):  # time
            fenzi += (variable[i][900] - z_hat) * (variable[i][j + 1] - pj[j])
            fenmu1 += (variable[i][900] - z_hat) ** 2
            fenmu2 += (variable[i][j + 1] - pj[j]) ** 2
        fenmu = (fenmu1 * fenmu2) ** 0.5
        if fenmu == 0:
            m.append(0)
        else:
            m.append(fenzi / fenmu)
    m = np.array(m)
    return m


def exponential(cc, D):
    sita = []
    for j in range(len(D)):
        sita.append(np.exp(-cc[j] * D[j]))
    return sita

def origin(cc,D):
    sita = []
    for j in range(len(D)):
        sita.append(1)
    return sita


def gaussian(cc, D):
    sita = []
    for j in range(len(D)):
        sita.append(np.exp(-(cc[j] * D[j]) ** 2))
    return sita


def linear(cc, D):
    sita = []
    for j in range(len(D)):
        if cc[j] < D[j]:
            a = 1
        else:
            a = 0
        if D[j] == 0:
            sita.append(1)
        else:
            sita.append(1 - (1 - cc[j] / D[j]) * a)
    return sita


def quadratic(cc, D):
    sita = []
    for j in range(len(D)):
        sita.append(1 / (1 + (cc[j] * D[j]) ** 2))
    return sita


def spherical(cc, D):
    sita = []
    for j in range(len(D)):
        if cc[j] < D[j]:
            a = 1
        else:
            a = 0
        if D[j] == 0:
            sita.append(1)
        else:
            sita.append(1 - (1 - 1.5 * cc[j] / D[j] + 0.5 * (cc[j] / D[j]) ** 3) * a)
    return sita


def extract_nc(path, coord_path, variable_name, precision=3):
    """extract variable(given region by coord) from .nc file
    input:
        path: path of the source nc file
        coord_path: path of the coord extracted by fishnet: OID_, lon, lat
        variable_name: name of the variable need to read
        precision: the minimum precision of lat/lon, to match the lat/lon of source nc file

    output:
        {variable_name}.txt [i, j]: i(file number) j(grid point number)
        lat_index.txt/lon_index.txt
        coord.txt
    """
    print(f"variable:{variable_name}")
    coord = pd.read_csv(coord_path, sep=",")  # read coord(extract by fishnet)
    print(f"grid point number:{len(coord)}")
    coord = coord.round(precision)  # 处理单位以便与nc中lat lon一致
    result = [path + "\\" + d for d in os.listdir(path) if d[-3:] == ".nc"]
    print(f"file number:{len(result)}")
    variable = np.zeros((len(result), len(coord) + 1))  # save the path correlated with read order

    # calculate the index of lat/lon in coord from source nc file
    f1 = Dataset(result[0], 'r')
    Dataset.set_auto_mask(f1, False)
    lat_index = []
    lon_index = []
    lat = f1.variables["lat"][:]
    lon = f1.variables["lon"][:]
    value = max(coord["lon"])
    x = coord["lon"].tolist()
    idx = x.index(value)
    y = coord["lat"][899]
    for j in range(len(coord)):
        lat_index.append(np.where(lat == coord["lat"][j])[0][0])
        lon_index.append(np.where(lon == coord["lon"][j])[0][0])
    f1.close()

    # read variable based on the lat_index/lon_index
    for i in range(len(result)):
        f = Dataset(result[i], 'r')
        Dataset.set_auto_mask(f, False)
        variable[i, 0] = float(re.search(r"\d{6}", result[i])[0])
        for j in range(len(coord)):
            variable[i, j + 1] = f.variables[variable_name][lat_index[j], lon_index[j]]
        # require: nc file only have three dimension
        # f.variables['Rainf_f_tavg'][0, lat_index_lp, lon_index_lp]is a mistake, we only need the file
        # that lat/lon corssed (1057) rather than meshgrid(lat, lon) (1057*1057)
        print(f"complete read file:{i}")
        f.close()

    # sort by time
    variable = variable[variable[:, 0].argsort()]
    target = [y, value]
    D = euclidean(coord["lat"], coord["lon"], target)
    m = cc(variable)
    sita = spherical(m, D)  # 切换核函数
    irrigation = np.arange(96)
    judge = []
    for j in range(len(D)):
        x = 0
        for i in range(96):
            # variable[i][j + 1] = variable[i][j + 1] * sita[j]
            x += variable[i][j + 1]
        judge.append(x)
    lat_new = []
    lon_new = []
    for j in range(len(D)):
        if judge[j] != 0:
            lat_new.append(lat[lat_index[j]])
            lon_new.append(lon[lon_index[j]])
    # for j in range(len(D)):
    #     if judge[j] != 0:
    #         irrigation=np.c_[irrigation, variable[:, [j+1]]]

    two = '000000000011000000110011100100110100000000001001100011100000101100011011010001110001000000100010001110111100010001101111101101111101110001110101101111111011100011011101111111101111110111111110011001000000010000000001100101101000001000010000110000000000010000000101000010100101000100010001010001000000000001111000001100000001010100001000000100111000100010101001110001000000001010000000000000000100111111111110101111110110111000011111101000001011011111110100110011111101000101011011101111111011111'
    two_list = list(two)
    lat_new_two = []
    lon_new_two = []
    for j in range(len(two_list)):
        if two_list[j] != '0':
            lat_new_two.append(lat_new[j])
            lon_new_two.append(lon_new[j])
    m = {'lat': lat_new_two, 'lon': lon_new_two}
    m = pd.DataFrame(m)
    m.to_excel(r'D:\Desktop\answer_sediment.xlsx')
    # irrigation = np.array(irrigation)
    # data = {'linear': irrigation}
    # df = DataFrame(data)
    # df.to_excel('C:\\Users\\Hasee\\Desktop\\new.xlsx')
    # print(irrigation)
    # sita = np.array(sita)
    # plt.figure(facecolor='w')
    # plt.plot(range(1, 97), irrigation, 'go', markersize=5)
    # plt.title('spherical', fontsize=16)
    # plt.xlabel('D')
    # plt.ylabel('spatial correlation')
    # plt.grid(True)
    # plt.show()

    # save
    # np.savetxt(f'{variable_name}.txt', variable, delimiter=' ')
    # np.savetxt('lat_index.txt', lat_index, delimiter=' ')
    # np.savetxt('lon_index.txt', lon_index, delimiter=' ')
    # coord.to_csv("coord.txt")
    return variable


def overview(path):
    # overview of the nc file
    result = [path + "/" + d for d in os.listdir(path) if d[-3:] == ".nc"]
    rootgrp = Dataset(result[0], "r")
    print('****************************')
    print(f"number of nc file:{len(result)}")
    print('****************************')
    print(f"variable key:{rootgrp.variables.keys()}")
    print('****************************')
    print(f"rootgrp:{rootgrp}")
    print('****************************')
    print(f"lat:{rootgrp.variables['lat'][:]}")
    print('****************************')
    print(f"lon:{rootgrp.variables['lon'][:]}")
    print(f"variable:{rootgrp.variables}")
    print('****************************')
    variable_name = input("variable name:")  # if you want to see the variable, input its name here
    while variable_name != "":
        print('****************************')
        print(f"variable:{rootgrp.variables[variable_name]}")
        variable_name = input("variable name:")  # if you want to quit, input enter here
    rootgrp.close()


if __name__ == "__main__":
    path = r"D:\Desktop\IWUEN细分"
    coord_path = r"D:\Desktop\coord.txt"
    effect = extract_nc(path, coord_path, "iwuen", precision=3)
    pd.DataFrame(effect[:, np.sum(effect, axis=0) != 0]).to_excel(r'C:\Users\Hasee\Desktop\yzsb.xlsx')
    print('hh')
