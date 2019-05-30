import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def read_data(file_path, header=None, names=None, usecols=None):
    """
    :param file_path:
    :param header:
    :param names:
    :param usecols:
    :return: matrix has m x n
    """
    df = pd.read_csv(file_path, header=header, names=names, usecols=usecols)
    # df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], format='%Y-%m-%d %H:%M:%S')
    df = df.dropna()
    # df.plot()
    # plt.show()
    return df.values


def difference(data, interval=1):
    """

    :param data: matrix m x n
    :param interval:
    :return: matrix (m-1) x n
    """
    diff = []
    for i in range(interval, len(data)):
        value = data[i] - data[i - interval]
        diff.append(value)
    return np.array(diff)


def inverse_difference(history, yhat, interval=1):
    """

    :param history: matrix m x n
    :param yhat: matrix (m-1) x n
    :param interval:
    :return:
    """
    return yhat + history[:-interval]


def log_diff(data, interval=1):
    """

    :param data: matrix m x n
    :param interval:
    :return: matrix (m-1) x n
    """
    log_diff = []
    for i in range(interval, len(data)):
        value = np.ma.log(data[i]) - np.ma.log(data[i - interval])
        log_diff.append(value)
    log_diff = np.array(log_diff)
    return log_diff


def inverse_log_diff(history, yhat, interval=1):
    return history[:-interval] * np.exp(yhat)


def scale(X):
    max = 1
    min = 0
    dmin = X.min(axis=0)
    dmax = X.max(axis=0)
    X_std = (X - dmin) / (dmax - dmin)
    X_scaled = X_std * (max - min) + min
    return X_scaled, dmin, dmax


def inverse_scale(data, dmin, dmax, feature=0):
    dmin = dmin[feature]
    dmax = dmax[feature]
    x_std = 0.5 * (data + 1)
    return x_std * (dmax - dmin) + dmin


def to_supervised(data, sliding):
    result = []
    n_exams = len(data)
    for i in range(sliding+1):
        result.append(data[i:n_exams-sliding+i])
    return np.concatenate(result, axis=1)



def prepare_data(file, header, names, usecols, sliding, test_size=0.2):
    data = read_data(file, header, names, usecols)
    data_scaled, dmin, dmax = scale(data)
    data_sup = to_supervised(data_scaled, sliding)
    data_sup = np.reshape(data_sup, (data_sup.shape[0], sliding+1, data.shape[-1]))
    x = data_sup[:, :-1, :]
    y = data_sup[:, -1, 0:1]

    n_exams = len(x)
    n_trains = int(n_exams * (1 - test_size))

    x_train = x[:n_trains]
    y_train = y[:n_trains]

    x_test = x[n_trains:]
    y_test = y[n_trains:]

    return (x_train, y_train), (x_test, y_test), (dmin, dmax)


# full_features = ["time_stamp", "numberOfTaskIndex", "numberOfMachineId",
#                          "meanCPUUsage", "canonical_memory_usage", "AssignMem",
#                          "unmapped_cache_usage", "page_cache_usage", "max_mem_usage",
#                          "mean_diskIO_time", "mean_local_disk_space", "max_cpu_usage",
#                          "max_disk_io_time", "cpi", "mai", "sampling_portion",
#                          "agg_type", "sampled_cpu_usage"]
# usecols = ['meanCPUUsage', 'canonical_memory_usage', 'AssignMem', 'unmapped_cache_usage', 'page_cache_usage',
#            'max_mem_usage', "max_cpu_usage"]
#
#
# usecols = ['meanCPUUsage', 'canonical_memory_usage']


#

class DataLoader():
    def __init__(self, filename, usecols, is_scale=False, is_log=False, is_diff=False):
        self.filename = filename
        self.usecols = usecols
        self.is_scale = is_scale
        self.is_log = is_log
        self.is_diff = is_diff

        self.raw_data = pd.read_csv(filename, usecols=usecols)

        # f = plt.figure(1)
        # self.raw_data.plot()
        # plt.title('raw')
        # f.show()

        self.data = self.raw_data.copy().values
        if is_log and is_diff:
            self.log_difference()
            # f = plt.figure(2)
            # df = pd.DataFrame(self.data)
            # df.plot()
            # plt.title('log_diff')
            # f.show()
        elif is_diff:
            self.difference()
            # f = plt.figure(3)
            # df = pd.DataFrame(self.data)
            # df.plot()
            # plt.title('diff')
            # f.show()
        if is_scale:
            self.scale()
        #     f = plt.figure(4)
        #     df = pd.DataFrame(self.data)
        #     df.plot()
        #     plt.title('scale')
        #     f.show()
        # plt.show()

    def scale(self, feature_range=(-1, 1)):
        self.feature_range = feature_range
        data = self.data
        dmin = data.min(axis=0)
        dmax = data.max(axis=0)
        self.dmin = dmin
        self.dmax = dmax
        data_std = (data- dmin) / (dmax - dmin)
        data_scaled = data_std * (feature_range[1] - feature_range[0]) + feature_range[0]
        self.data = data_scaled

    def inverse_scale(self, data_scaled, feature=[0]):
        dmin = self.dmin[feature]
        dmax = self.dmax[feature]
        feature_range = self.feature_range
        assert data_scaled.shape[1] == len(feature)
        data_std = (data_scaled - feature_range[0]) / (feature_range[1] - feature_range[0])
        raw_data = data_std * (dmax - dmin) + dmin
        return raw_data

    def difference(self, interval=1):
        """

        :param data: matrix m x n
        :param interval:
        :return: matrix (m-1) x n
        """
        data = self.data
        diff = []
        for i in range(interval, len(data)):
            value = data[i] - data[i - interval]
            diff.append(value)
        self.data = np.array(diff)

    def inverse_difference(self, data, history, interval=1):
        """

        :param history: matrix m x n
        :param yhat: matrix (m-1) x n
        :param interval:
        :return:
        """
        return data + history[:-interval]

    def log_difference(self, interval=1):
        """

        :param data: matrix m x n
        :param interval:
        :return: matrix (m-1) x n
        """
        data = self.data
        log_diff = []
        for i in range(interval, len(data)):
            value = np.ma.log(data[i]) - np.ma.log(data[i - interval])
            log_diff.append(value)
        log_diff = np.array(log_diff)
        self.data = log_diff

    def inverse_log_difference(self, data, interval=1):
        history = self.raw_data.values[self.n_train-interval:-(self.n_in + self.n_out + interval)]
        return history[:-interval] * np.exp(data)

    def to_supervised(self, data, n_in, n_out):
        result = []
        n_exams = len(data)
        for i in range(n_in + n_out):
            result.append(data[i:n_exams - (n_in + n_out) + i])
        return np.concatenate(result, axis=1)

    def get_data(self, n_in, n_out, split_rate=0.8):
        data = self.data
        self.n_in = n_in
        self.n_out = n_out
        data_sup = self.to_supervised(data, n_in, n_out)
        data_sup = np.reshape(data_sup, (data_sup.shape[0], n_in + n_out, -1))
        n_train = int(len(data) * split_rate)
        self.n_train = n_train
        data_sup_train = data_sup[:n_train]
        data_sup_test = data_sup[n_train:]
        xe_train = data_sup_train[:, :n_in, :]
        xd_train = data_sup_train[:, n_in - n_out:n_in, :]
        yd_train = data_sup_train[:, n_in:, :1]
        y_train = data_sup_train[:, n_in:, 0]

        xe_test = data_sup_test[:, :n_in, :]
        xd_test = data_sup_test[:, n_in - n_out:n_in, :]
        yd_test = data_sup_test[:, n_in:, :1]
        y_test = data_sup_test[:, n_in:, 0]
        return ([xe_train, xd_train], yd_train, y_train), ([xe_test, xd_test], yd_test, y_test)





# dataloader = DataLoader('datasets/wc98/wc98_workload_5min.csv', usecols=[1], is_diff=True, is_log=True, is_scale=True)
# raw_data = dataloader.raw_data.copy().values
# print(raw_data[:5])
# print(dataloader.data[:5])
#
# inverted_data = dataloader.inverse_log_difference(data=dataloader.data, history=dataloader.raw_data.values)
# print(inverted_data[:5])


