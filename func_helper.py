#!/usr/bin/env python
# coding=utf-8
'''
 * @File    :  func_helper.py
 * @Time    :  2020/09/25 22:06:38
 * @Author  :  Hanielxx
 * @Version :  1.0
 * @Desc    :  数据处理及辅助函数
'''
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def max_min_scaler(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def get_mid(data):
    # data: series
    # idx在n为偶数时返回n/2
    x = data.sort_values()
    n = x.shape[0]
    idx = (n + 1) / 2 if n % 2 == 1 else n / 2
    idx = int(idx - 1)
    mid = x.iloc[idx] if n % 2 == 1 else (x.iloc[idx] + x.iloc[int((n + 2) / 2 - 1)]) / 2
    return idx, mid


def get_F(data, low_coef=1.5, up_coef=1.5):
    # data: series
    x = data.sort_values()
    n = x.shape[0]
    idx_2, q_2 = get_mid(x)
    k = n // 4
    # print(k, n)
    if n % 2 == 0:
        q_1_data = x.iloc[:idx_2] if idx_2 % 2 else x.iloc[:idx_2 + 1]
        q_3_data = x.iloc[idx_2 + 1:]
        idx_1, q_1 = get_mid(q_1_data)
        idx_3, q_3 = get_mid(q_3_data)
    elif n % 4 == 3:
        q_1 = 0.75 * x.iloc[k] + 0.25 * x.iloc[k + 1]
        q_3 = 0.25 * x.iloc[3 * k] + 0.75 * x.iloc[3 * k + 2]
    elif n % 4 == 1:
        q_1 = 0.25 * x.iloc[k - 1] + 0.75 * x.iloc[k]
        q_3 = 0.75 * x.iloc[3 * k] + 0.25 * x.iloc[3 * k + 1]
    i_qr = q_3 - q_1
    f_1, f_u = q_1 - low_coef * i_qr, q_3 + up_coef * i_qr
    return f_1, f_u


def horizonal_process(current_data, low_coef=1.5, up_coef=1.5):
    '''传入DataFrame，返回异常数据的index
    '''
    res_data = current_data[current_data['label'] != 1].copy()
    res_data = res_data.sort_values(by=['Power'])
    st = np.linspace(0, 2000, 81, dtype=int)

    # 对每个power段
    for i in range(1, st.shape[0]):
        p_min, p_max = st[i - 1], st[i]
        flag = (res_data['Power'] >= p_min) & (res_data['Power'] <= p_max)
        sub_data = res_data[flag]

        if sub_data.shape[0] < 4:
            # continue
            f_1 = f_u = 0
        else:
            f_1, f_u = get_F(sub_data['WindSpeed'], low_coef, up_coef)

        flag_1_u = (res_data['WindSpeed'] < f_1) | (res_data['WindSpeed'] > f_u)
        flag_com = flag & flag_1_u
        res_data.loc[flag_com, 'label'] = 1

    # 返回DataFrame和下标
    res_data.loc[res_data[res_data['label'] == 1].index, 'label'] = 1
    return res_data, res_data[res_data['label'] == 1].index


def vertical_process(current_data, low_coef=1.5, up_coef=1.5):
    res_data = current_data[current_data['label'] != 1].copy()
    res_data = res_data.sort_values(by=['Power'])
    st = np.arange(0, 25.5, 0.5)

    # 对每个风速段
    for i in range(1, st.shape[0]):
        p_min, p_max = st[i - 1], st[i]
        flag = (res_data['WindSpeed'] >= p_min) & (res_data['WindSpeed'] <= p_max)
        sub_data = res_data[flag]
        # print(sub_data)

        if sub_data.shape[0] < 4:
            # continue
            f_1 = f_u = 0
        else:
            f_1, f_u = get_F(sub_data['Power'], low_coef, up_coef)

        # 超过[F1, Fu]的都算作异常
        flag_1_u = (res_data['Power'] < f_1) | (res_data['Power'] > f_u)
        flag_com = flag & flag_1_u
        res_data.loc[flag_com, 'label'] = 1

    # 返回DataFrame和下标
    res_data.loc[res_data[res_data['label'] == 1].index, 'label'] = 1
    return res_data, res_data[res_data['label'] == 1].index


def get_cut_int_out(current_rotor):
    '''返回当前风电机切入和切出风速
    '''
    cut_int = cut_out = 0
    if current_rotor in [1, 2, 3, 4, 6, 7, 8, 9, 10]:
        cut_in = 3
        cut_out = 25
    elif current_rotor == 5:
        cut_in = 3
        cut_out = 22
    elif current_rotor == 11:
        cut_in = 2.5
        cut_out = 19
    elif current_rotor == 12:
        cut_in = 3
        cut_out = 22
    return cut_in, cut_out


def judge_data(row, cut_in, cut_out):
    # 剩下功率小于等于0,都是异常
    if row['Power'] < 0:
        return 1
    # 额定风速以内的,功率为0的,为异常
    elif row['Power'] == 0 and row['WindSpeed'] >= cut_in and row['WindSpeed'] <= cut_out:
        return 1
    # # 风机转速小于0,或者大于max时为异常,没考虑转速范围的最小值
    elif row['RotorSpeed'] < 0 or row['RotorSpeed'] > row['WheelSpeedMax']:
        # elif row['RotorSpeed'] < 0:
        return 1
    # 风速小于0为异常
    elif row['WindSpeed'] < 0:
        return 1
    # 切出风速以上，power大于0的为异常
    elif row['Power'] > 0 and row['WindSpeed'] > cut_out:
        return 1
    # 功率除以转速平方超过10为异常
    elif row['Power'] > 0 and row['Power'] / row['RotorSpeed'] / row['RotorSpeed'] > 10:
        return 1
    else:
        return row['label']


def remove_neg_data(current_data, current_rotor):
    '''返回current_data功率异常区域为负数的下标
    '''
    current_data = current_data[current_data['label'] != 1].copy()
    cut_in, cut_out = get_cut_int_out(current_rotor)
    tmp_data = current_data

    tmp_data['label'] = tmp_data.apply(judge_data, axis=1, args=(cut_in, cut_out))

    return tmp_data, tmp_data[tmp_data['label'] == 1].index


def linear_regression(x, y):
    '''拟合 y=ax+b
    '''
    linreg = LinearRegression()
    linreg.fit(x, y)
    res = linreg.coef_
    return linreg.coef_[0], linreg.intercept_


def remove_mid_anomaly(current_data, max_a=0.5, max_diff=3, max_time=12, min_speed=0.3, power_step=6):
    '''传入的是待处理的风电机数据，在函数内部对功率进行划分
    max_a = 0.5 # y=ax+b，这个设置基本没用，都是接近0的
    max_diff = 3 # 3kw
    max_time = 12 # 6*10s
    min_speed = 0.3 # 0.5m/s
    '''
    res_data = current_data[current_data['label'] != 1].copy()
    st = np.arange(0, 1900, power_step)

    res_idx = np.array([])

    for i in range(1, st.shape[0]):
        p_min, p_max = st[i - 1], st[i]
        flag = (res_data['Power'] >= p_min) & (res_data['Power'] <= p_max)
        sub_data = res_data[flag]

        # 拟合y=ax+b
        x = sub_data['WindSpeed']
        y = sub_data['Power']
        if (x.shape[0] == 0):
            continue
        a, b = np.polyfit(x, y, 1)
        # print("{}-{}: {}个点， y={:.2f}x+{:.2f}".format(p_min, p_max, x.shape[0], a, b))

        # 处理符合条件的点
        # if(a>max_a):
        #     continue

        # 和拟合曲线差在max_diff以内
        sub_data['diff'] = np.abs(sub_data['WindSpeed'] * a + b - sub_data['Power'])
        diff_idx = sub_data[sub_data['diff'] < max_diff].index
        tmp_idx = []

        # diff_idx[0]忽略
        for i in range(1, diff_idx.shape[0]):
            cur_idx, pre_idx = diff_idx[i], diff_idx[i - 1]
            cur_windspeed, pre_windspeed = sub_data.loc[cur_idx, 'WindSpeed'], sub_data.loc[pre_idx, 'WindSpeed']

            # print("idx: {}-{}, diff_idx: {}, diff_speed: {}, diff_power: {}".format(pre_idx, cur_idx,cur_idx-pre_idx, cur_windspeed-pre_windspeed, sub_data.loc[cur_idx, 'Power']-sub_data.loc[pre_idx, 'Power']))
            # 时间差在max_time以内，且windspeed差在min_speed以上
            if (cur_idx - pre_idx < max_time and cur_windspeed - pre_windspeed > min_speed):
                tmp_idx.append(cur_idx)
                # print("idx: {}-{}, diff_idx: {}, diff_speed: {}, diff_power: {}".format(pre_idx, cur_idx,cur_idx-pre_idx, cur_windspeed-pre_windspeed, sub_data.loc[cur_idx, 'Power']-sub_data.loc[pre_idx, 'Power']))

        tmp_idx = np.array(tmp_idx)
        res_idx = np.concatenate((res_idx, tmp_idx))

    res_data.loc[res_idx, 'label'] = 1
    return res_data, res_idx


def plot_current_data(current_data, raw_data, rotor_num, save=False, file_path=None, plot=True):
    plt.figure(figsize=(24, 7))

    plt.subplot(141)
    g_data = raw_data[raw_data['WindNumber'] == rotor_num]
    plt.plot(g_data['WindSpeed'], g_data['Power'], '.y', ms=4, label='origin data')
    plt.ylim((-100, 2200))
    plt.xlim((-1, 27))
    plt.legend()

    plt.subplot(142)
    g_data = current_data
    plt.plot(g_data['WindSpeed'], g_data['Power'], '.g', ms=4, label='origin data')
    plt.ylim((-100, 2200))
    plt.xlim((-1, 27))
    plt.legend()

    plt.subplot(143)
    g_data = current_data.loc[current_data['label'] != 1]
    plt.plot(g_data['WindSpeed'], g_data['Power'], '.b', ms=4, label='normal data')
    plt.ylim((-100, 2200))
    plt.xlim((-1, 27))
    plt.legend()

    plt.subplot(144)
    g_data = current_data.loc[current_data['label'] == 1]
    plt.plot(g_data['WindSpeed'], g_data['Power'], '.r', ms=4, label='anomaly data')
    plt.ylim((-100, 2200))
    plt.xlim((-1, 27))
    plt.legend()

    if save and file_path is not None:
        plt.savefig(file_path)
    if plot:
        plt.show()


def process_current_data(data,
                         current_rotor=None,
                         remove_neg=True,
                         remove_mid=True,
                         remove_horizonal=True,
                         remove_vertical=True,
                         horizonal_low=1.5,
                         horizonal_up=1.5,
                         vertical_low=1.5,
                         vertical_up=1.5,
                         max_a=0.5,
                         max_diff=3,
                         max_time=12,
                         min_speed=0.3,
                         power_step=6):
    current_data = data.copy()

    # 去底部异常
    if remove_neg:
        removed_neg_data, neg_power_idx = remove_neg_power(current_data, current_rotor)
        current_data.loc[neg_power_idx, 'label'] = 1
        print("rotor {}, bottom: {}".format(current_rotor, neg_power_idx.shape[0]))

    # 去中部异常
    if (remove_mid):
        removed_mid_data, mid_anomaly_idx = remove_mid_anomaly(current_data, max_a, max_diff, max_time, min_speed,
                                                               power_step)
        current_data.loc[mid_anomaly_idx, 'label'] = 1
        print("rotor {}, mid: {}".format(current_rotor, mid_anomaly_idx.shape[0]))

    # 横向四分法
    if remove_horizonal:
        horizonal_data, horizonal_anomaly_idx = horizonal_process(current_data, horizonal_low, horizonal_up)
        current_data.loc[horizonal_anomaly_idx, 'label'] = 1
        print("rotor {}, horizonal: {}".format(current_rotor, horizonal_anomaly_idx.shape[0]))

    # 纵向四分法
    if remove_vertical:
        vertical_data, vertical_anomaly_idx = vertical_process(current_data, vertical_low, vertical_up)
        current_data.loc[vertical_anomaly_idx, 'label'] = 1
        print("rotor {}, vertical: {}".format(current_rotor, vertical_anomaly_idx.shape[0]))

    return current_data, current_data[current_data['label'] == 1].index


def update_raw(raw_data, idx, val):
    # 更新到raw_data
    # current_anomaly_idx = current_data[current_data['label'] == 1].index
    print("sum of {} data to update: {}".format(val, idx.shape[0]))

    raw_data.loc[idx, 'label'] = val
    print("sum of {} in raw data: {}".format(val, np.sum(raw_data['label'] == val)))


def process_raw_data(raw_data):
    raw_data['label'] = -1
    raw_data['Year'] = raw_data['Time'].apply(lambda x: int(x.split(' ')[0].split('/')[0]))
    raw_data['Month'] = raw_data['Time'].apply(lambda x: int(x.split(' ')[0].split('/')[1]))
    raw_data['Day'] = raw_data['Time'].apply(lambda x: int(x.split(' ')[0].split('/')[2]))
    raw_data['Index'] = raw_data.index

    # 把windspeed和power归一化保存
    raw_data['NormWindSpeed'] = raw_data[['WindSpeed']].apply(max_min_scaler)
    raw_data['NormPower'] = raw_data[['Power']].apply(max_min_scaler)

    # raw_data = raw_data[[
    #     'WindNumber', 'Time', 'WindSpeed', 'Power', 'RotorSpeed', 'label', 'Year', 'Month', 'Day', 'NormWindSpeed',
    #     'NormPower'
    # ]]
    return raw_data
