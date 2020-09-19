#!/usr/bin/env python
# coding=utf-8
'''
 * @File    :  data_process.py
 * @Time    :  2020/09/18 19:36:00
 * @Author  :  Hanielxx
 * @Version :  1.0
 * @Desc    :  处理风电机数据
'''
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data_path = './data/combined_data.csv'
FIG_PATH = './figures4'
RES_PATH = './submission/submission_4.csv'


def judge_power_min(row):
    # 切入以下的且功率小于0的暂且认为正常
    if row['Power'] < 0 and row['WindSpeed'] < cut_in_99:
        return 0
    # 切入以下功率大于0的为异常
    elif row['Power'] >= 0 and row['WindSpeed'] < cut_in_99:
        return 1
    # 切入风速以上，切出风速以下，power小于等于0的为异常
    elif row['Power'] <= 0 and row['WindSpeed'] < cut_out_99 and row[
            'WindSpeed'] > cut_in_99:
        return 1
    # 切出风速以上，power大于0的为异常
    elif row['Power'] > 0 and row['WindSpeed'] > cut_out_99:
        return 1
    # 切出风速以上，power小于0的暂且认为是正常
    elif row['Power'] <= 0 and row['WindSpeed'] > cut_out_99:
        return 0
    else:
        return -1


def get_cut_in_out(current_rotor):
    if current_rotor in [1, 2, 3, 4, 6, 7, 8, 9, 10]:
        cut_in_99 = 3
        cut_out_99 = 25
    elif current_rotor == 5:
        cut_in_99 = 3
        cut_out_99 = 22
    elif current_rotor == 11:
        cut_in_99 = 2.5
        cut_out_99 = 19
    elif current_rotor == 12:
        cut_in_99 = 3
        cut_out_99 = 22
    return cut_in_99, cut_out_99


def get_mid(data):
    # data: series
    # idx在n为偶数时返回n/2
    x = data.sort_values()
    n = x.shape[0]
    idx = (n + 1) / 2 if n % 2 == 1 else n / 2
    idx = int(idx - 1)
    mid = x.iloc[idx] if n % 2 == 1 else (x.iloc[idx] +
                                          x.iloc[int((n + 2) / 2 - 1)]) / 2
    return idx, mid


def get_F(data):
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
    # f_1, f_u = q_1 - 1.5 * i_qr, q_3 + 1.5 * i_qr
    f_1, f_u = q_1 - 1.6 * i_qr, q_3 + 1.6 * i_qr
    return f_1, f_u


if __name__ == "__main__":
    raw_data = pd.read_csv(data_path)
    raw_data['label'] = -1
    for current_rotor in range(1, 13):
        print("当前处理： rotor {}".format(current_rotor))
        current_data = raw_data[raw_data['WindNumber'] == current_rotor]

        plt.figure(figsize=(25, 10))
        g_data = current_data[current_data['label'] != 1]
        plt.plot(g_data['WindSpeed'],
                 g_data['Power'],
                 '.b',
                 ms=4,
                 label='WindSpeed-Power')
        plt.legend()
        plt.xlabel('Wind Speed')
        plt.ylabel('Power')
        cur_title = "{}_origin".format(current_rotor)
        plt.title(cur_title)
        plt.savefig('{}/{}.png'.format(FIG_PATH, cur_title))
        # plt.show()

        cut_in_99, cut_out_99 = get_cut_in_out(current_rotor)

        current_data['label'] = current_data[['Power', 'WindSpeed'
                                              ]].apply(judge_power_min, axis=1)

        # 处理横向四分
        horizonal_data = current_data[current_data['label'] != 1]
        horizonal_data = horizonal_data.sort_values(by=['Power'])

        st_horizonal = np.linspace(0, 2000, 81, dtype=int)

        # 超过[F1, Fu]的都算作异常
        for i in range(1, st_horizonal.shape[0]):
            p_min, p_max = st_horizonal[i - 1], st_horizonal[i]
            flag = (horizonal_data['Power'] >=
                    p_min) & (horizonal_data['Power'] <= p_max)
            sub_data = horizonal_data[flag]
            if sub_data.shape[0] < 4:
                # continue
                f_1 = f_u = 0
            else:
                f_1, f_u = get_F(sub_data['WindSpeed'])
            flag_1_u = (horizonal_data['WindSpeed'] <
                        f_1) | (horizonal_data['WindSpeed'] > f_u)
            flag_com = flag & flag_1_u
            horizonal_data.loc[flag_com, 'label'] = 1

        current_data.loc[horizonal_data[horizonal_data['label'] == 1].index,
                         'label'] = 1

        plt.figure(figsize=(25, 10))
        g_data_0 = current_data[current_data['label'] == 0]
        plt.plot(g_data_0['WindSpeed'],
                 g_data_0['Power'],
                 '.b',
                 ms=4,
                 label='normal data')
        g_data_1 = current_data[current_data['label'] == 1]
        plt.plot(g_data_1['WindSpeed'],
                 g_data_1['Power'],
                 '.r',
                 ms=4,
                 label='abnormal data')
        g_data_2 = current_data[current_data['label'] == -1]
        plt.plot(g_data_2['WindSpeed'],
                 g_data_2['Power'],
                 '.y',
                 ms=4,
                 label='others')
        plt.legend()
        plt.xlabel('Wind Speed')
        plt.ylabel('Power')
        cur_title = "{}_horizonal".format(current_rotor)
        plt.title(cur_title)
        plt.savefig('{}/{}.png'.format(FIG_PATH, cur_title))
        # plt.show()

        if current_rotor not in [6]:
            # 纵向四分
            vertical_data = current_data[current_data['label'] != 1]
            vertical_data = vertical_data.sort_values(by=['Power'])

            st_vertical = np.arange(0, 25.5, 0.5)

            # 超过[F1, Fu]的都算作异常
            for i in range(1, st_vertical.shape[0]):
                p_min, p_max = st_vertical[i - 1], st_vertical[i]
                flag = (vertical_data['WindSpeed'] >=
                        p_min) & (vertical_data['WindSpeed'] <= p_max)
                sub_data = vertical_data[flag]
                # print(sub_data)
                if sub_data.shape[0] < 4:
                    # continue
                    f_1 = f_u = 0
                else:
                    f_1, f_u = get_F(sub_data['Power'])
                flag_1_u = (vertical_data['Power'] <
                            f_1) | (vertical_data['Power'] > f_u)
                flag_com = flag & flag_1_u
                vertical_data.loc[flag_com, 'label'] = 1

            current_data.loc[vertical_data[vertical_data['label'] == 1].index,
                             'label'] = 1

            plt.figure(figsize=(25, 10))
            g_data_0 = current_data[current_data['label'] == 0]
            plt.plot(g_data_0['WindSpeed'],
                     g_data_0['Power'],
                     '.b',
                     ms=4,
                     label='normal data')
            g_data_1 = current_data[current_data['label'] == 1]
            plt.plot(g_data_1['WindSpeed'],
                     g_data_1['Power'],
                     '.r',
                     ms=4,
                     label='abnormal data')
            g_data_2 = current_data[current_data['label'] == -1]
            plt.plot(g_data_2['WindSpeed'],
                     g_data_2['Power'],
                     '.y',
                     ms=4,
                     label='others')
            plt.legend()
            plt.xlabel('Wind Speed')
            plt.ylabel('Power')
            cur_title = "{}_horizonal_vertical".format(current_rotor)
            plt.title(cur_title)
            plt.savefig('{}/{}.png'.format(FIG_PATH, cur_title))
            # plt.show()

        current_data.loc[current_data['label'] == -1, 'label'] = 0

        cur_idx = current_data.index
        raw_data.loc[current_data.index, 'label'] = current_data['label']

    print("未分类数：{}".format(np.sum(raw_data['label'] == -1)))
    print("异常点数：{}".format(np.sum(raw_data['label'] == 1)))
    print("正常点数：{}".format(np.sum(raw_data['label'] == 0)))
    result = raw_data[['WindNumber', 'Time', 'label']]
    result.to_csv(RES_PATH, index=None)
