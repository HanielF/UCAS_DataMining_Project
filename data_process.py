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
import time


def plot_all(data, rotor_start, rotor_end):
    '''绘制所有风机的图像
    '''
    for i in range(rotor_start, rotor_end + 1):
        current_rotor = i
        current_data = data[data['WindNumber'] == current_rotor]
        current_data, idx = process_current_data(current_data,
                                                 current_rotor=i,
                                                 remove_neg=True,
                                                 remove_mid=False,
                                                 remove_horizonal=False,
                                                 remove_vertical=False)
        data.loc[idx, 'label'] = 1
        line_num = ''
        is_save = False
        # plot_current_data(current_data, rotor_num=current_rotor, save=is_save)


if __name__ == "__main__":
    log_path = './log'
    data_path = './data/combined_data.csv'
    suffix = "9_29_3"
    fig_path = './figures/submission_{}/'.format(suffix)
    res_path = './submission/submission_{}.csv'.format(suffix)
    neg_data_path = './data/add_cutin_neg_data_label.csv'
    is_save = False
    is_plot = False
    anomaly_val = 1

    current_time = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())
    log("\n\n==> Time: " + current_time)

    # 读入数据并进行一些特征处理
    raw_data = pd.read_csv(data_path)
    raw_data = process_raw_data(raw_data)

    # 预处理negative data label
    # plot_all(raw_data, 1, 12)
    # raw_data[['label']].to_csv(neg_data_path)

    # 读入预处理的negative data label
    neg_label = pd.read_csv(neg_data_path)
    raw_data['label'] = neg_label['label']
    log("Init neg data size: {}".format(np.sum(neg_label['label'] == 1)))
    before_sub = pd.read_csv('./submission/submission_9_29.csv')
    # neg_label = pd.read_csv('./data/remove2k_neg_data_label.csv')
    # print("last neg label: {}".format(np.sum(neg_label['label'] == 1)))
    # test_data = raw_data.loc[raw_data['label'] != neg_label['label']]
    # plot_all(test_data, 1, 12)

    # 一号
    current_rotor = 1
    current_data = raw_data[raw_data['WindNumber'] == current_rotor]
    # plot_current_data(current_data, rotor_num=current_rotor, save=False)

    current_data, idx = process_current_data(current_data,
                                             current_rotor,
                                             horizonal_low=2.5,
                                             horizonal_up=2.5,
                                             vertical_low=2.5,
                                             vertical_up=2.5,
                                             kmeans=True,
                                             linear=True,
                                             linear_outlier=0.01,
                                             linear_max_x=25)

    line_num = 1
    fp = fig_path + "{}_{}.png".format(current_rotor, line_num)
    plot_current_data(current_data, rotor_num=current_rotor, save=is_save, file_path=fp, plot=is_plot)
    update_raw(raw_data, idx, anomaly_val)
    log('before submission: {}'.format(np.sum(before_sub['WindNumber'] == current_rotor & (before_sub['label'] == 1))))

    # 2号
    current_rotor = 2
    line_num = 1
    fp = fig_path + "{}_{}.png".format(current_rotor, line_num)

    current_data = raw_data[raw_data['WindNumber'] == current_rotor]
    current_data, idx = process_current_data(current_data,
                                             current_rotor,
                                             horizonal_low=2,
                                             horizonal_up=2,
                                             vertical_low=2,
                                             kmeans=True,
                                             linear=True,
                                             linear_outlier=0.05)

    plot_current_data(current_data,
                      raw_data=raw_data,
                      rotor_num=current_rotor,
                      save=is_save,
                      file_path=fp,
                      plot=is_plot)
    update_raw(raw_data, idx, anomaly_val)

    # 3号
    current_rotor = 3
    # 处理第二条较低的线,作为异常
    current_data = raw_data[raw_data['WindNumber'] == current_rotor]

    oct29_data = current_data[((current_data['Month'] == 10) & (current_data['Day'] == 29))][:99]
    oct30_data = current_data[((current_data['Month'] == 10) & (current_data['Day'] == 30))][50:]

    line1_data = current_data[(current_data['Month'] == 1) | (current_data['Month'] == 2) | (current_data['Month'] == 3)
                              | (current_data['Month'] == 9) | (current_data['Month'] == 11) |
                              (current_data['Month'] == 12) | ((current_data['Month'] == 4) &
                                                               (current_data['Day'] < 12)) |
                              ((current_data['Month'] == 8) & (current_data['Day'] > 21)) |
                              ((current_data['Month'] == 10) & (current_data['Day'] < 29))]
    line1_data = pd.concat([line1_data, oct29_data, oct30_data])
    current_data.loc[line1_data.index, 'label'] = 1
    update_raw(raw_data, line1_data.index, anomaly_val)

    # 处理第一条线 5,6,7月，4月18-31号，8月1-21号
    oct29_data = current_data[((current_data['Month'] == 10) & (current_data['Day'] == 29))][99:]
    oct30_data = current_data[((current_data['Month'] == 10) & (current_data['Day'] == 30))][:50]

    current_data = current_data[(current_data['Month'] == 5) | (current_data['Month'] == 6) |
                                (current_data['Month'] == 7) | ((current_data['Month'] == 4) &
                                                                (current_data['Day'] >= 12)) |
                                ((current_data['Month'] == 8) & (current_data['Day'] <= 21))]
    current_data = pd.concat([current_data, oct29_data, oct30_data])

    current_data, idx = process_current_data(current_data,
                                             current_rotor,
                                             horizonal_low=2,
                                             horizonal_up=2,
                                             vertical_low=2,
                                             vertical_up=2,
                                             kmeans=True,
                                             linear=True,
                                             linear_outlier=0.15)

    line_num = 1
    fp = fig_path + "{}_{}.png".format(current_rotor, line_num)
    plot_current_data(current_data, rotor_num=current_rotor, save=is_save, file_path=fp, plot=is_plot)
    update_raw(raw_data, idx, anomaly_val)

    # 四号
    current_rotor = 4
    current_data = raw_data[raw_data['WindNumber'] == current_rotor]

    current_data, idx = process_current_data(current_data,
                                             current_rotor,
                                             horizonal_low=2.5,
                                             horizonal_up=2.5,
                                             vertical_low=2.5,
                                             vertical_up=2.5,
                                             kmeans=True,
                                             linear=True,
                                             linear_outlier=0.1)

    line_num = 1
    fp = fig_path + "{}_{}.png".format(current_rotor, line_num)
    plot_current_data(current_data, rotor_num=current_rotor, save=is_save, file_path=fp, plot=is_plot)
    update_raw(raw_data, idx, anomaly_val)

    # 五号
    current_rotor = 5
    current_data = raw_data[raw_data['WindNumber'] == current_rotor]

    current_data, idx = process_current_data(
        current_data,
        current_rotor,
        horizonal_low=2.5,
        horizonal_up=2.5,
        remove_vertical=False,
        # vertical_low=2.5,
        # vertical_up=1.5,
        kmeans=False,
        linear=True,
        linear_outlier=0.01)

    line_num = 1
    fp = fig_path + "{}_{}.png".format(current_rotor, line_num)
    plot_current_data(current_data, rotor_num=current_rotor, save=is_save, file_path=fp, plot=is_plot)
    update_raw(raw_data, idx, anomaly_val)

    # 六号
    current_rotor = 6
    # 处理第二条线 5月18-31号，6，7，8月
    current_data = raw_data[raw_data['WindNumber'] == current_rotor]
    line_2 = current_data[(current_data['Month'] == 6) | (current_data['Month'] == 7) | (current_data['Month'] == 8) |
                          ((current_data['Month'] == 5) & (current_data['Day'] >= 18))]

    line_3 = current_data[(current_data['Month'] == 1) | (current_data['Month'] == 2) | (current_data['Month'] == 3) |
                          (current_data['Month'] == 4) | (current_data['Month'] == 11) | (current_data['Month'] == 12) |
                          ((current_data['Month'] == 5) & (current_data['Day'] < 18))]

    current_data = pd.concat([line_2, line_3])
    update_raw(raw_data, current_data.index, anomaly_val)

    # 处理第一条线 9，10月
    current_data = raw_data[raw_data['WindNumber'] == current_rotor]
    current_data = current_data[(current_data['Month'] == 9) | (current_data['Month'] == 10)]
    current_data, idx = process_current_data(current_data,
                                             current_rotor,
                                             horizonal_low=2,
                                             horizonal_up=2,
                                             vertical_low=2,
                                             vertical_up=2,
                                             kmeans=False,
                                             linear=True,
                                             linear_outlier=0.1)

    line_num = 1
    fp = fig_path + "{}_{}.png".format(current_rotor, line_num)
    current_data = raw_data[raw_data['WindNumber'] == current_rotor]
    plot_current_data(current_data, rotor_num=current_rotor, save=is_save, file_path=fp, plot=is_plot)
    update_raw(raw_data, idx, anomaly_val)

    # 七号
    current_rotor = 7
    # 处理左边，9月10-17
    current_data = raw_data[raw_data['WindNumber'] == current_rotor]
    line_left = current_data[((current_data['Month'] == 9) & (current_data['Day'] <= 17) & (current_data['Day'] >= 10))]
    current_data.loc[line_left.index, 'label'] = 1

    # 处理右边
    current_data = current_data[current_data['label'] != 1]
    current_data, idx = process_current_data(current_data,
                                             current_rotor,
                                             horizonal_low=2,
                                             horizonal_up=2,
                                             vertical_low=2,
                                             vertical_up=2,
                                             kmeans=False,
                                             linear=True,
                                             linear_outlier=0.05)

    line_num = 1
    fp = fig_path + "{}_{}.png".format(current_rotor, line_num)
    plot_current_data(current_data, rotor_num=current_rotor, save=is_save, file_path=fp, plot=is_plot)
    update_raw(raw_data, idx, anomaly_val)

    # 八号
    current_rotor = 8
    current_data = raw_data[raw_data['WindNumber'] == current_rotor]

    current_data, idx = process_current_data(current_data,
                                             current_rotor,
                                             horizonal_low=2.5,
                                             horizonal_up=2.5,
                                             vertical_low=2.5,
                                             vertical_up=2.5,
                                             kmeans=True,
                                             linear=True,
                                             linear_outlier=0.05)

    line_num = 1
    fp = fig_path + "{}_{}.png".format(current_rotor, line_num)
    plot_current_data(current_data, rotor_num=current_rotor, save=is_save, file_path=fp, plot=is_plot)
    update_raw(raw_data, idx, anomaly_val)

    # 九号
    current_rotor = 9
    current_data = raw_data[raw_data['WindNumber'] == current_rotor]
    # 处理下面曲线
    current_data = current_data[((current_data['Month'] == 1) | (current_data['Month'] == 2) |
                                 (current_data['Month'] == 3) | (current_data['Month'] == 4) |
                                 (current_data['Month'] == 5) | (current_data['Month'] == 6) |
                                 (current_data['Month'] == 7) | (current_data['Month'] == 11) |
                                 (current_data['Month'] == 12) | ((current_data['Month'] == 8) &
                                                                  (current_data['Day'] <= 26)))]
    current_data, idx = process_current_data(current_data,
                                             current_rotor,
                                             horizonal_low=2.5,
                                             horizonal_up=2.5,
                                             vertical_low=2.5,
                                             vertical_up=2.5,
                                             kmeans=True,
                                             linear=True,
                                             linear_outlier=0.1)
    update_raw(raw_data, idx, anomaly_val)

    # 处理左上的线
    current_data = raw_data[raw_data['WindNumber'] == current_rotor]
    current_data = current_data[((current_data['Month'] == 9) | (current_data['Month'] == 10) |
                                 ((current_data['Month'] == 8) & (current_data['Day'] > 26)))]
    current_data, idx = process_current_data(current_data,
                                             current_rotor,
                                             horizonal_low=2.5,
                                             horizonal_up=2.5,
                                             vertical_low=2.5,
                                             vertical_up=2.5,
                                             kmeans=False,
                                             linear=True,
                                             linear_outlier=0.1)

    line_num = 1
    fp = fig_path + "{}_{}.png".format(current_rotor, line_num)
    current_data = raw_data[raw_data['WindNumber'] == current_rotor]
    plot_current_data(current_data, rotor_num=current_rotor, save=is_save, file_path=fp, plot=is_plot)
    update_raw(raw_data, idx, anomaly_val)

    # 十号
    current_rotor = 10
    current_data = raw_data[raw_data['WindNumber'] == current_rotor]
    line_left = current_data[((current_data['Month'] == 12) | ((current_data['Month'] == 1) &
                                                               (current_data['Day'] <= 7)) |
                              ((current_data['Month'] == 11) & (current_data['Day'] >= 12)))]
    current_data.loc[line_left.index, 'label'] = 1
    update_raw(raw_data, line_left.index, anomaly_val)
    current_data, idx = process_current_data(current_data,
                                             current_rotor,
                                             horizonal_low=2,
                                             horizonal_up=2,
                                             vertical_low=2,
                                             vertical_up=2,
                                             kmeans=False,
                                             linear=True,
                                             linear_outlier=0.1)

    line_num = 1
    fp = fig_path + "{}_{}.png".format(current_rotor, line_num)
    plot_current_data(current_data, rotor_num=current_rotor, save=is_save, file_path=fp, plot=is_plot)
    update_raw(raw_data, idx, anomaly_val)

    # 十一号
    current_rotor = 11
    current_data = raw_data[raw_data['WindNumber'] == current_rotor]

    current_data, idx = process_current_data(current_data,
                                             current_rotor,
                                             horizonal_low=1.5,
                                             horizonal_up=1.5,
                                             vertical_low=1.5,
                                             vertical_up=1.5,
                                             kmeans=True,
                                             linear=False,
                                             linear_outlier=0.02)

    line_num = 1
    fp = fig_path + "{}_{}.png".format(current_rotor, line_num)
    plot_current_data(current_data, rotor_num=current_rotor, save=is_save, file_path=fp, plot=is_plot)
    update_raw(raw_data, idx, anomaly_val)

    # 十二号
    current_rotor = 12
    current_data = raw_data[raw_data['WindNumber'] == current_rotor]

    current_data, idx = process_current_data(current_data,
                                             current_rotor,
                                             horizonal_low=2.5,
                                             horizonal_up=2.5,
                                             vertical_low=2.5,
                                             vertical_up=2.5,
                                             kmeans=False,
                                             linear=True,
                                             linear_outlier=0.05)

    line_num = 1
    fp = fig_path + "{}_{}.png".format(current_rotor, line_num)
    plot_current_data(current_data, rotor_num=current_rotor, save=is_save, file_path=fp, plot=is_plot)
    update_raw(raw_data, idx, anomaly_val)

    # 保存结果
    print("总数据量: {}".format(raw_data.shape[0]))
    print("异常点数：{}".format(np.sum(raw_data['label'] == 1)))
    print("正常点数：{}".format(np.sum(raw_data['label'] == 0)))

    if is_save:
        result = raw_data[['WindNumber', 'Time', 'label']]
        result.to_csv(res_path, index=None)

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


def plot_all(data, rotor_start, rotor_end):
    '''绘制所有风机的图像
    '''
    for i in range(rotor_start, rotor_end + 1):
        current_rotor = i
        current_data = data[data['WindNumber'] == current_rotor]
        current_data, idx = process_current_data(current_data,
                                                 current_rotor=i,
                                                 remove_neg=True,
                                                 remove_mid=False,
                                                 remove_horizonal=False,
                                                 remove_vertical=False)
        data.loc[idx, 'label'] = 1
        line_num = ''
        is_save = False
        # plot_current_data(current_data, rotor_num=current_rotor, save=is_save)
