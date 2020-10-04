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
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler


def max_min_scaler(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def log(s, fp='./log'):
    print(s)
    if fp is not None:
        with open(fp, 'a') as f:
            f.write(s + '\n')


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
    # log(k, n)
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
        # log(sub_data)

        if sub_data.shape[0] < 4:
            # continue
            f_1 = f_u = 0
        else:
            f_1, f_u = get_F(sub_data['Power'], low_coef, up_coef)

        # 超过[F1, Fu]的都算作异常
        # flag_1_u = (res_data['Power'] < f_1) | (res_data['Power'] > f_u)
        # 只管在f_u下面的,超过f_u的算异常
        flag_1_u = res_data['Power'] > f_u
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
    # 剩下功率小于等于0,都是异常, 测试功率超过额定功率是否算异常
    # 是否设置等于0还有待验证
    if row['Power'] <= 0:
        return 1
    # 额定风速以内的,功率为0的,为异常
    elif row['Power'] == 0 and row['WindSpeed'] >= cut_in and row['WindSpeed'] <= cut_out:
        return 1
    # 额定风速以外的, 功率不为0的,为异常
    # elif row['Power'] > 0 and (row['WindSpeed'] < cut_in or row['WindSpeed'] > cut_out):
    elif row['Power'] > 0 and (row['WindSpeed'] > cut_out):
        return 1
    # # 风机转速小于0,或者大于max时为异常,没考虑转速范围的最小值
    # elif row['RotorSpeed'] < 0 or row['RotorSpeed'] > row['WheelSpeedMax']:
    elif row['RotorSpeed'] < 0:
        return 1
    # 风速小于0为异常
    # elif row['WindSpeed'] < 0 or row['WindSpeed'] > cut_out:
    elif row['WindSpeed'] < 0:
        return 1
    # 功率除以转速平方超过10为异常
    # elif row['Power'] > 0 and row['Power'] / row['RotorSpeed'] / row['RotorSpeed'] > 10:
    #     return 1
    else:
        return row['label']


def align_vertical(current_data, step=10):
    '''按照垂直方向对齐，返回对齐后的数据，这里对原数据进行了修改
    '''
    current_data["OldWindSpeed"] = current_data.WindSpeed
    bins = pd.cut(current_data.Power, np.arange(-1000, 3000, step))
    current_data["hbins"] = bins
    groups = []
    for name, group in current_data.groupby("hbins"):
        if np.sum(group.label == 0) == 0:
            groups.append(group)
            continue
        group.WindSpeed -= np.quantile(group.WindSpeed[group.label == 0], 0.05)
        groups.append(group)
    # 按照下标排序
    current_data = pd.concat(groups, sort=False)
    current_data = current_data.sort_index()
    return current_data


def remove_area(row, rotor_num):
    '''针对不同的风电机,去除特定区域的数据
    '''
    if row['label'] == 1:
        return 1
    if rotor_num == 3 or rotor_num == 4:
        return 0
    if rotor_num == 1 and row['WindSpeed'] > 8 and row['Power'] > 460 and row['Power'] < 480:
        return 1
    elif rotor_num == 1 and row['WindSpeed'] > 13 and row['Power'] > 1650 and row['Power'] < 1950:
        return 1
    elif rotor_num == 2 and row['WindSpeed'] > 9 and row['Power'] > 630 and row['Power'] < 650:
        return 1
    elif rotor_num == 2 and row['WindSpeed'] > 10.5 and row['Power'] > 1000 and row['Power'] < 1075:
        return 1
    elif rotor_num == 2 and row['WindSpeed'] > 13 and row['Power'] > 1650 and row['Power'] < 1900:
        return 1
    elif rotor_num == 5 and row['WindSpeed'] > 8.6 and row['Power'] > 900 and row['Power'] < 950:
        return 1
    elif rotor_num == 5 and row['WindSpeed'] > 8.1 and row['Power'] > 700 and row['Power'] < 740:
        return 1
    elif rotor_num == 5 and row['WindSpeed'] > 11.5 and row['Power'] > 1540 and row['Power'] < 1560:
        return 1
    elif rotor_num == 6 and row['WindSpeed'] > 9 and row['Power'] > 300 and row['Power'] < 1100:
        return 1
    elif rotor_num == 7 and row['WindSpeed'] > 7.5 and row['Power'] > 430 and row['Power'] < 475:
        return 1
    elif rotor_num == 7 and row['WindSpeed'] > 8 and row['Power'] > 100 and row['Power'] < 600:
        return 1
    elif rotor_num == 7 and row['WindSpeed'] > 11 and row['Power'] > 1200 and row['Power'] < 1300:
        return 1
    elif rotor_num == 8 and row['WindSpeed'] > 12.7 and row['Power'] > 1700 and row['Power'] < 1950:
        return 1
    elif rotor_num == 8 and row['WindSpeed'] > 7.9 and row['Power'] > 0 and row['Power'] < 600:
        return 1
    elif rotor_num == 9 and row['WindSpeed'] > 11 and row['Power'] > 1640 and row['Power'] < 1750:
        return 1
    elif rotor_num == 9 and row['WindSpeed'] > 10.5 and row['Power'] > 200 and row['Power'] < 600:
        return 1
    elif rotor_num == 9 and row['WindSpeed'] > 12 and row['Power'] > 1200 and row['Power'] < 1700:
        return 1
    elif rotor_num == 10 and row['WindSpeed'] > 7.5 and row['Power'] > 150 and row['Power'] < 500:
        return 1
    elif rotor_num == 11 and row['WindSpeed'] > 12 and row['Power'] > 1 and row['Power'] < 1770:
        return 1
    elif rotor_num == 12 and row['WindSpeed'] > 7.3 and row['Power'] > 0 and row['Power'] < 740:
        return 1
    elif rotor_num == 12 and row['WindSpeed'] > 5.3 and row['Power'] > 180 and row['Power'] < 300:
        return 1
    else:
        return 0


def remove_neg_data(current_data, current_rotor):
    '''返回current_data功率异常区域为负数的下标
    '''
    current_data = current_data[current_data['label'] != 1].copy()
    cut_in, cut_out = get_cut_int_out(current_rotor)
    tmp_data = current_data

    tmp_data['label'] = tmp_data.apply(judge_data, axis=1, args=(cut_in, cut_out))
    tmp_data['label'] = tmp_data.apply(remove_area, axis=1, args=(current_rotor, ))

    return tmp_data, tmp_data[tmp_data['label'] == 1].index


def remove_mid_anomaly(current_data, max_a=0.5, max_diff=3, max_time=6, min_speed=0.3, power_step=6):
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
        # log("{}-{}: {}个点， y={:.2f}x+{:.2f}".format(p_min, p_max, x.shape[0], a, b))

        # 处理符合条件的点
        if (a > max_a):
            continue

        # 和拟合曲线差在max_diff以内
        sub_data['diff'] = np.abs(sub_data['WindSpeed'] * a + b - sub_data['Power'])
        diff_idx = sub_data[sub_data['diff'] < max_diff].index
        tmp_idx = []

        # diff_idx[0]忽略
        for i in range(1, diff_idx.shape[0]):
            cur_idx, pre_idx = diff_idx[i], diff_idx[i - 1]
            cur_windspeed, pre_windspeed = sub_data.loc[cur_idx, 'WindSpeed'], sub_data.loc[pre_idx, 'WindSpeed']

            # 时间差在max_time以内，且windspeed差在min_speed以上
            if (cur_idx - pre_idx < max_time and cur_windspeed - pre_windspeed > min_speed):
                tmp_idx.append(cur_idx)
                # log("idx: {}-{}, diff_idx: {}, diff_speed: {}, diff_power: {}".format(pre_idx, cur_idx,cur_idx-pre_idx, cur_windspeed-pre_windspeed, sub_data.loc[cur_idx, 'Power']-sub_data.loc[pre_idx, 'Power']))

        tmp_idx = np.array(tmp_idx)
        res_idx = np.concatenate((res_idx, tmp_idx))

    res_data.loc[res_idx, 'label'] = 1
    return res_data, res_idx


def plot_current_data(current_data, rotor_num, raw_data=None, save=False, file_path=None, plot=True):
    plt.figure(figsize=(30, 7))

    plt.subplot(141)
    g_data = current_data.loc[current_data['label'] != 1]
    plt.plot(g_data['WindSpeed'], g_data['Power'], '.b', ms=4, label='normal data')
    plt.ylim((-100, 2200))
    plt.xlim((-1, 27))
    plt.xticks(np.arange(-1, 28))
    plt.legend()

    plt.subplot(142)
    g_data = current_data.loc[current_data['label'] == 1]
    plt.plot(g_data['WindSpeed'], g_data['Power'], '.r', ms=4, label='anomaly data')
    plt.ylim((-100, 2200))
    plt.xlim((-1, 27))
    plt.legend()

    plt.subplot(143)
    g_data = current_data
    plt.plot(g_data['WindSpeed'], g_data['Power'], '.g', ms=4, label='current data')
    plt.ylim((-100, 2200))
    plt.xlim((-1, 27))
    plt.legend()

    if raw_data is not None:
        plt.subplot(144)
        g_data = raw_data[raw_data['WindNumber'] == rotor_num]
        plt.plot(g_data['WindSpeed'], g_data['Power'], '.y', ms=4, label='rotor data')
        plt.ylim((-100, 2200))
        plt.xlim((-1, 27))
        plt.legend()

    if save and file_path is not None:
        plt.savefig(file_path)
    if plot:
        plt.show()


def update_raw(raw_data, idx, val):
    # 更新到raw_data
    log("sum of {} data to update: {}".format(val, idx.shape[0]))

    raw_data.loc[idx, 'label'] = val
    log("sum of {} in raw data: {}\n".format(val, np.sum(raw_data['label'] == val)))


def process_raw_data(raw_data, columns=None):
    # 初始化label
    raw_data['label'] = 0
    # 添加year mon day index特征
    raw_data['Year'] = raw_data['Time'].apply(lambda x: int(x.split(' ')[0].split('/')[0]))
    raw_data['Month'] = raw_data['Time'].apply(lambda x: int(x.split(' ')[0].split('/')[1]))
    raw_data['Day'] = raw_data['Time'].apply(lambda x: int(x.split(' ')[0].split('/')[2]))
    raw_data['Index'] = raw_data.index

    # 把windspeed和power归一化保存
    # raw_data['NormWindSpeed'] = raw_data[['WindSpeed']].apply(max_min_scaler)
    # raw_data['NormPower'] = raw_data[['Power']].apply(max_min_scaler)

    if columns is not None:
        raw_data = raw_data[columns]
    return raw_data


def linear_data(data, x='WindSpeed', y='Power', step=1, max_x=25, outliers_fraction=0.1):
    ''' 局部线性回归
    '''
    cur_data = data[data['label'] != 1].copy()
    cur_data.loc[:, 'Power'] = cur_data['Power'] / 50.0

    num = int(max_x / step)
    for i in range(num):
        start, end = i * step, (i + 1) * step
        # log("start: {}, end: {}".format(start, end))

        # 获得每个分组内的数据
        data = cur_data.loc[(cur_data[x] >= start) & (cur_data[x] < end)]
        # log(data.head())
        data = data.sort_values(by=y)
        data = data[[x, y]]

        interval = data.shape[0]
        if interval < 10:
            cur_data.loc[data.index, 'label'] = 1
            continue

        # 选取四分位数作为筛异常点的标准
        if i < 3:
            outliers_fraction += 0.1
        residual = np.quantile(np.abs(data[y] - np.median(data[y])), 1 - outliers_fraction, interpolation='midpoint')
        ransac = RANSACRegressor(LinearRegression(),
                                 max_trials=100,
                                 min_samples=0.5,
                                 random_state=0,
                                 residual_threshold=residual,
                                 loss='squared_loss')
        ransac.fit(data[[x]], data[[y]])
        inlier_mask = ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)

        # 可视化
        # plt.figure(figsize=(20, 16))
        # plt.scatter(data.loc[inlier_mask, x], data.loc[inlier_mask, y], c='blue', marker='o', label='Inliers')
        # plt.scatter(data.loc[outlier_mask, x], data.loc[outlier_mask, y], c='lightgreen', marker='s', label='Outliers')

        # line_x = np.arange(int(data[x].min() - 1), int(data[x].max() + 2), 1)
        # line_y = ransac.predict(line_x[:, np.newaxis])
        # plt.plot(line_x, line_y, color='red', lw=4)
        # plt.show()

        # 更新数据
        cur_data.loc[data[outlier_mask].index, 'label'] = 1
        cur_data.loc[data[inlier_mask].index, 'label'] = 0

    return cur_data, cur_data[cur_data['label'] == 1].index


def svm_data(data, x='WindSpeed', y='Power', step=0.5, max_x=25, outliers_fraction=0.2):
    ''' 局部svm行不通,会把中间值去掉
    '''
    cur_data = data[data['label'] != 1].copy()

    num = int(max_x / step)
    for i in range(num):
        start, end = i * step, (i + 1) * step
        # log("start: {}, end: {}".format(start, end))

        #获得每个分组内的数据
        data = cur_data.loc[(cur_data[x] >= start) & (cur_data[x] < end)]
        # log(data.head())
        data = data.sort_values(by=y)
        data = data[[x, y]]

        interval = data.shape[0]
        if interval < 7:
            cur_data.loc[data.index, 'label'] = 1
            continue

        scaler = StandardScaler()
        trans_data = pd.DataFrame(scaler.fit_transform(data))
        svm = OneClassSVM(nu=outliers_fraction, kernel='rbf')
        svm.fit(trans_data)

        data['label'] = svm.predict(trans_data)
        # svm返回1或者-1,表示正常和异常

        data['label'] = data.apply(lambda row: 0 if row['label'] == 1 else 1, axis=1)
        # 更新数据
        cur_data.loc[data[data['label'] == 1].index, 'label'] = 1

        # 可视化
        # plt.scatter(data[x], data[y], c=data['label'])
        # plt.show()

    return cur_data, cur_data[cur_data['label'] == 1].index


def kmeans_data(df, Pe=25, ratio=0.5, T=400):
    result_df = df[0:0]
    delete_df = df[0:0]
    result_df['label'] = 0
    delete_df['label'] = 1
    #获得分组数量,按照风速从0-25分50组
    num = int(Pe / ratio)
    #关键字
    key = 'Power'
    key_sort = 'WindSpeed'
    #按分组大小对数据进行划分，并处理每个分组
    for i in range(num):
        start = i * ratio
        end = (i + 1) * ratio
        # log("start: {}, end: {}".format(start, end))
        #获得每个分组内的数据
        data = df[df[key_sort] >= start]
        data = data[data[key_sort] < end]
        # 按照power排序
        data = data.sort_values(by=key)
        interval = len(data)

        if interval < 5:
            data['label'] = 0
            delete_df = delete_df.append(data)
            continue

        # power, windspeed
        loan = np.array(data[[key, key_sort]])
        clf = KMeans(n_clusters=5, random_state=9)
        clf = clf.fit(loan)
        data['label'] = clf.labels_
        w_max = max(clf.cluster_centers_[:, 0])
        for index, w in enumerate(clf.cluster_centers_[:, 0]):
            if w_max - w <= T:
                result_df = result_df.append(data[data['label'] == index])
            else:
                delete_df = delete_df.append(data[data['label'] == index])
        # # 可视化
        # plt.scatter(loan[:, 1], loan[:, 0], c=clf.labels_)
        # plt.show()

    result_df['label'] = 0
    delete_df['label'] = 1
    return result_df.append(delete_df), delete_df.index


# 修改后的k-means聚类，效果不太好
def kmeans_data_modified(df, Pe=25, ratio=0.5, T=400):
    result_df = df[0:0]
    delete_df = df[0:0]
    result_df['label'] = 0
    delete_df['label'] = 1
    #获得分组数量,按照风速从0-25分50组
    num = int(Pe / ratio)
    #关键字
    key = 'Power'
    key_sort = 'WindSpeed'
    #按分组大小对数据进行划分，并处理每个分组
    for i in range(num):
        start = i * ratio
        end = (i + 1) * ratio
        # log("start: {}, end: {}".format(start, end))
        #获得每个分组内的数据
        data = df[df[key_sort] >= start]
        data = data[data[key_sort] < end]
        # 按照power排序
        data = data.sort_values(by=key)
        interval = len(data)

        if interval < 7:
            data['label'] = 0
            delete_df = delete_df.append(data)
            continue

        # power, windspeed
        loan = data[[key, key_sort]].values
        clf = KMeans(n_clusters=7, random_state=9).fit(loan)

        data['label'] = clf.labels_
        center_w = clf.cluster_centers_[:, 0]
        w_idx = np.argsort(center_w)
        w_max, w_min = center_w[w_idx[-1]], center_w[w_idx[0]]
        min_cnt = int(data.shape[0] / 21)  # 平均数的1/3
        # 处理最大和最小的簇
        if (w_max - center_w[w_idx[1]] > T):
            log("聚类,max{}簇头{:.2f}距离太远,删除".format(w_idx[-1], w_max))
            delete_df = delete_df.append(data[data['label'] == index])
            w_max = center_w[w_idx[-2]]
        if (center_w[w_idx[-2]] - w_min > T):
            log("聚类,min{}簇头{:.2f}距离太远,删除".format(w_idx[0], w_min))
            delete_df = delete_df.append(data[data['label'] == index])
            w_min = center_w[w_idx[1]]

        for index, w in enumerate(center_w):
            # 如果个数小于7,直接去除
            if (np.sum(data['label'] == index) <= 6):
                log("聚类,第{}簇头{:.2f}个数{:.2f}少于7,删除".format(index, w, np.sum(data['label'] == index), min_cnt))
                delete_df = delete_df.append(data[data['label'] == index])
                continue
            # 上面两个和下面两个如果个数太少也去掉
            if (index in [w_idx[0], w_idx[1], w_idx[-1], w_idx[-2]] and np.sum(data['label'] == index) <= min_cnt):
                log("聚类,第{}簇头{:.2f}个数{:.2f}少于min_cnt={},删除".format(index, w, np.sum(data['label'] == index), min_cnt))
                delete_df = delete_df.append(data[data['label'] == index])
                continue
            # power差在T=400以内
            if w_max - w <= T and w - w_min <= T:
                result_df = result_df.append(data[data['label'] == index])
            else:
                log("聚类,内部簇{}的中心{:.2f}距离max/min簇头{:.2f}/{:.2f}太远,删除".format(index, w, w_max, w_min))
                delete_df = delete_df.append(data[data['label'] == index])

        # # 可视化
        # plt.scatter(loan[:, 1], loan[:, 0], c=clf.labels_)
        # plt.show()

    result_df['label'] = 0
    delete_df['label'] = 1
    return result_df.append(delete_df), delete_df.index


def dbscan_data(data, x='WindSpeed', y='Power', step=0.5, max_x=25, dbscan_eps=40, dbscan_samples=5):
    ''' 局部dbscan
    '''
    cur_data = data[data['label'] != 1].copy()
    cur_data.loc[:, 'Power'] = cur_data['Power'] / 50.0

    num = int(max_x / step)
    dbscan = DBSCAN(dbscan_eps, dbscan_samples)

    for i in range(num):
        start, end = i * step, (i + 1) * step
        log("start: {}, end: {}".format(start, end))

        # 获得每个分组内的数据
        data = cur_data.loc[(cur_data[x] >= start) & (cur_data[x] < end)]
        # log(data.head())
        data = data.sort_values(by=y)
        data = data[[x, y]]

        # 每组个数少于5个就跳过
        interval = data.shape[0]
        if interval < 5:
            cur_data.loc[data.index, 'label'] = 0
            continue

        label = dbscan.fit_predict(data[[x, y]].values)
        data["cls"] = label
        clsmean = data.groupby("cls").apply(lambda x: np.mean(x.Power))
        if len(clsmean) != 1:
            clsmean = clsmean[clsmean.index != -1]
        maxcls = clsmean.sort_values().index[-1]
        data["label"] = [0 if lab == maxcls else 1 for lab in label]
        data = data.drop(columns=["cls"])

        # 可视化
        plt.scatter(data[x], data[y], c=label)
        plt.show()

        # 更新数据
        cur_data.loc[data[data['label']==1].index, 'label'] = 1

    return cur_data, cur_data[cur_data['label'] == 1].index


def align_remove_data(current_data, step=0.3):
    '''在数据对齐后，去除比0.9*max小的数据
    '''
    df = current_data[current_data['label'] != 1].copy()
    # 按照windspeed划分
    bins = pd.cut(df.WindSpeed, np.arange(-15, 30, step))
    df["vbins"] = bins
    groups = []
    for name, group in df.groupby("vbins"):
        if group.shape[0] == 0:
            continue
        if name.left > 2:
            group.loc[group.Power < np.max(df.Power) * 0.9, "label"] = 1
        groups.append(group)
    df = pd.concat(groups, sort=False).sort_index()
    return df, df[df['label'] == 1].index


def process_current_data(data,
                         current_rotor=None,
                         remove_neg=False,
                         remove_horizonal=True,
                         horizonal_low=1.5,
                         horizonal_up=1.5,
                         remove_vertical=True,
                         vertical_low=1.5,
                         vertical_up=1.5,
                         remove_mid=True,
                         max_a=0.5,
                         max_diff=3,
                         max_time=12,
                         min_speed=0.3,
                         power_step=6,
                         align=False,
                         align_remove=False,
                         kmeans=False,
                         svm=False,
                         linear=False,
                         linear_outlier=0.1,
                         linear_max_x=25,
                         dbscan=False):
    current_data = data.copy()

    # 去底部异常
    if remove_neg:
        removed_neg_data, neg_power_idx = remove_neg_data(current_data, current_rotor)
        current_data.loc[neg_power_idx, 'label'] = 1
        log("rotor {}, bottom, 剩余正常点: {}, 检测出异常点: {}".format(current_rotor, np.sum(current_data['label'] != 1),
                                                             neg_power_idx.shape[0]))

    # 去中部异常
    if (remove_mid):
        removed_mid_data, mid_anomaly_idx = remove_mid_anomaly(current_data, max_a, max_diff, max_time, min_speed,
                                                               power_step)
        current_data.loc[mid_anomaly_idx, 'label'] = 1
        log("rotor {}, mid, 剩余正常点: {}, 检测出异常点: {}".format(current_rotor, np.sum(current_data['label'] != 1),
                                                          mid_anomaly_idx.shape[0]))

    # 横向四分法
    if remove_horizonal:
        horizonal_data, horizonal_anomaly_idx = horizonal_process(current_data, horizonal_low, horizonal_up)
        current_data.loc[horizonal_anomaly_idx, 'label'] = 1
        log("rotor {}, horizonal, 剩余正常点: {}, 检测出异常点: {}".format(current_rotor, np.sum(current_data['label'] != 1),
                                                                horizonal_anomaly_idx.shape[0]))

    # 纵向四分法
    if remove_vertical:
        vertical_data, vertical_anomaly_idx = vertical_process(current_data, vertical_low, vertical_up)
        current_data.loc[vertical_anomaly_idx, 'label'] = 1
        log("rotor {}, vertical, 剩余正常点: {}, 检测出异常点: {}".format(current_rotor, np.sum(current_data['label'] != 1),
                                                               vertical_anomaly_idx.shape[0]))

    # 对齐，从这开始数据和原始数据不一样，windspeed大小发生变化
    if align:
        current_data = align_vertical(current_data)

    # 对齐后去除比0.9×max小的数据
    if align_remove:
        align_remove_res, align_remove_idx = align_remove_data(current_data)
        current_data.loc[align_remove_idx, 'label'] = 1
        log("rotor {}, align remove, 剩余正常点: {}, 检测出异常点: {}".format(current_rotor, np.sum(current_data['label'] != 1),
                                                                   align_remove_idx.shape[0]))
    plot_current_data(current_data, current_rotor)

    # cluster, 只传正常数据
    if kmeans:
        cluster_res, cluster_idx = kmeans_data(current_data[current_data['label'] != 1], Pe=25, ratio=0.5, T=600)
        current_data.loc[cluster_idx, 'label'] = 1
        log("rotor {}, k_means, 剩余正常点: {}, 检测出异常点: {}".format(current_rotor, np.sum(current_data['label'] != 1),
                                                              cluster_idx.shape[0]))

    # dbscan, 只传正常数据
    if dbscan:
        dbscan_res, dbscan_idx = dbscan_data(current_data[current_data['label'] != 1])
        current_data.loc[dbscan_idx, 'label'] = 1
        log("rotor {}, k_means, 剩余正常点: {}, 检测出异常点: {}".format(current_rotor, np.sum(current_data['label'] != 1),
                                                              cluster_idx.shape[0]))

    # svm,只传正常数据
    if svm:
        cluster_res, cluster_idx = svm_data(current_data[current_data['label'] != 1])
        current_data.loc[cluster_idx, 'label'] = 1
        log("rotor {}, svm, 剩余正常点: {}, 检测出异常点: {}".format(current_rotor, np.sum(current_data['label'] != 1),
                                                          cluster_idx.shape[0]))

    # linear_regression
    if linear:
        linear_res, linear_idx = linear_data(current_data[current_data['label'] != 1],
                                             max_x=linear_max_x,
                                             outliers_fraction=linear_outlier)
        current_data.loc[linear_idx, 'label'] = 1
        log("rotor {}, linear, 剩余正常点: {}, 检测出异常点: {}".format(current_rotor, np.sum(current_data['label'] != 1),
                                                             linear_idx.shape[0]))

    return current_data, current_data[current_data['label'] == 1].index
