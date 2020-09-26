#!/usr/bin/env python
# coding=utf-8
'''
 * @File    :  data_process_2.py
 * @Time    :  2020/09/22 16:44:41
 * @Author  :  Hanielxx
 * @Version :  2.0
 * @Desc    :  处理风点击数据v2
'''

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from func_helper import *


data_path = './data/combined_data.csv'
raw_data = pd.read_csv(data_path)


def process_rotor2(data, current_rotor=2):
    # 去底部异常
    current_data = data.copy()
    removed_neg_data, neg_power_idx = remove_neg_power(current_data, current_rotor)
    current_data.loc[neg_power_idx, 'label'] = 1
    print("rotor 2, bottom: {}".format(neg_power_idx.shape))

    # 去除最下面的线
    low_idx = current_data[((current_data['Month'] == 1) | (current_data['Month'] == 2)
                            | (current_data['Month'] == 12))
                           & (current_data['WindSpeed'] > 9)].index
    current_data.loc[low_idx, 'label'] = 1
    print("rotor 2, low line: {}".format(low_idx.shape))

    # 去中部异常
    removed_mid_data, mid_anomaly_idx = remove_mid_anomaly(current_data, max_time=9, max_diff=8, power_step=20)
    current_data.loc[mid_anomaly_idx, 'label'] = 1
    print("rotor 2, mid: {}".format(mid_anomaly_idx.shape))

    # 横向四分法
    horizonal_data, horizonal_anomaly_idx = horizonal_process(current_data, low_coef=2.5, up_coef=2.8)
    current_data.loc[horizonal_anomaly_idx, 'label'] = 1
    print("rotor 2, horizonal: {}".format(horizonal_anomaly_idx.shape))

    # # 纵向四分法
    vertical_data, vertical_anomaly_idx = vertical_process(current_data[current_data['WindSpeed'] > 8])
    current_data.loc[vertical_anomaly_idx, 'label'] = 1
    print("rotor 2, vertical: {}".format(vertical_anomaly_idx.shape))

    # 横向四分法
    horizonal_data, horizonal_anomaly_idx = horizonal_process(current_data, low_coef=2, up_coef=1)
    current_data.loc[horizonal_anomaly_idx, 'label'] = 1
    print("rotor 2, horizonal: {}".format(horizonal_anomaly_idx.shape))

    return current_data, current_data[current_data['label'] == 1].index


if __name__ == "__main__":
    raw_data = process_raw_data(raw_data)

    is_save = False
    anomaly_val = 1

    # 一号
    current_rotor = 1
    current_data = raw_data[raw_data['WindNumber'] == current_rotor]

    current_data, idx = process_current_data(current_data, current_rotor=1)
    update_raw(raw_data, idx, anomaly_val)

    line_num = 1
    fp = "./figures/vp_{}_{}.png".format(current_rotor, line_num)
    plot_current_data(current_data, raw_data=raw_data, rotor_num=current_rotor, save=is_save, file_path=fp)

    # 2号
    current_rotor = 2
    current_data = raw_data[raw_data['WindNumber'] == current_rotor]

    # 单独处理
    current_data, idx = process_rotor2(current_data)
    update_raw(raw_data, idx, anomaly_val)

    line_num = 1
    fp = "./figures/vp_{}_{}.png".format(current_rotor, line_num)
    plot_current_data(current_data, raw_data=raw_data, rotor_num=current_rotor, save=is_save, file_path=fp)

    # 3号
    current_rotor = 3
    # 处理第一条线 5,6,7月，4月18-31号，8月1-21号
    current_data = raw_data[raw_data['WindNumber'] == current_rotor]

    oct29_data = current_data[((current_data['Month'] == 10) & (current_data['Day'] == 29))][99:]
    oct30_data = current_data[((current_data['Month'] == 10) & (current_data['Day'] == 30))][:50]

    current_data = current_data[(current_data['Month'] == 5) | (current_data['Month'] == 6) |
                                (current_data['Month'] == 7) | ((current_data['Month'] == 4) &
                                                                (current_data['Day'] >= 12)) |
                                ((current_data['Month'] == 8) & (current_data['Day'] <= 21))]
    current_data = pd.concat([current_data, oct29_data, oct30_data])

    # 寻找异常
    current_data, idx = process_current_data(
        current_data,
        current_rotor,
        remove_mid=False,
        vertical_low=2,
        vertical_up=2,
    )
    # 更新
    update_raw(raw_data, idx, anomaly_val)

    fp = "./figures/vp_{}_{}.png".format(current_rotor, line_num)
    plot_current_data(current_data, raw_data=raw_data, rotor_num=current_rotor, save=is_save, file_path=fp)
