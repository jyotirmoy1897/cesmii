from random_forest import RandomForest
from random import randrange
import numpy as np
import pandas as pd
from regression_tree import createForeCast
import timeit


class Partitions:
    def __init__(self, x, y, n_devices, n_estimators, ops, threshold, test_interval, sampling_factor, pred_threshold):
        self.data = x
        self.tag = y
        self.n_devices = n_devices
        self.partition_size = int(len(self.tag)/self.n_devices)
        self.n_estimators = n_estimators
        self.ops = ops
        self.threshold = threshold
        self.test_interval = test_interval
        self.sampling_factor = sampling_factor
        self.server_rf = None
        self.device_rf = {}
        self.pred_threshold = pred_threshold

    def rf_sampling(self, rf_list):
        n = len(rf_list)
        rf_sampled = []
        for i in range(int(n/self.sampling_factor)):
            sample_idx = np.random.choice(n, randrange(1, n + 1), replace=True)
            rf_sampled.append(rf_list[sample_idx[0]])
        return rf_sampled

    def rf_precict(self, rf_list, x):
        sum_pred = np.zeros(x.shape[0])
        count = 0
        for reg_tree in rf_list:
            y_pred = createForeCast(reg_tree, x)
            sum_pred += y_pred
            count += 1
        return sum_pred / count

    def find_min_distance(self, machine):
        return machine+1

    def independent_learning(self):
        rf_list = []
        for i in range(self.n_devices):
            if i != self.n_devices - 1:
                data_x = self.data[i * self.partition_size:(i+1)*self.partition_size, :]
                data_y = self.tag[i * self.partition_size:(i+1)*self.partition_size]
            else:
                data_x = self.data[i * self.partition_size:, :]
                data_y = self.tag[i * self.partition_size:]
            rf = RandomForest(n_estimators=self.n_estimators, ops=self.ops, threshold=self.threshold,
                              test_interval=self.test_interval)
            print("\nfit device_{num}".format(num=i))
            rf.fit(x=data_x, y=data_y)
            rf_list += rf.reg_trees
            self.device_rf[i] = rf
        self.server_rf = self.rf_sampling(rf_list)

    def shared_learning(self):
        rf_list = []
        for i in range(self.n_devices):
            if i != self.n_devices - 1:
                data_x = self.data[i * self.partition_size:(i + 1) * self.partition_size, :]
                data_y = self.tag[i * self.partition_size:(i + 1) * self.partition_size]
            else:
                data_x = self.data[i * self.partition_size:, :]
                data_y = self.tag[i * self.partition_size:]
            rf = RandomForest(n_estimators=self.n_estimators, ops=self.ops, threshold=self.threshold,
                              test_interval=self.test_interval)
            print("\nfit device_{num}".format(num=i))
            rf.fit(x=data_x, y=data_y)
            rf_list += rf.reg_trees
            self.device_rf[i] = rf
        server_rf = RandomForest(n_estimators=self.n_estimators, ops=self.ops, threshold=self.threshold,
                                 test_interval=self.test_interval)
        print("\nfit server")
        server_rf.fit(x=self.data, y=self.tag)
        self.server_rf = self.rf_sampling(rf_list) + server_rf.reg_trees

    def global_inferencing(self, x):
        input_machine = 0
        var_pred, y_pred = self.device_rf[input_machine].predict(x=x)
        if var_pred > self.pred_threshold:
            print("passing server")
            y_pred = self.rf_precict(rf_list=self.server_rf, x=x)
        return y_pred

    def local_inferencing(self, x):
        input_machine = 0
        var_pred, y_pred = self.device_rf[input_machine].predict(x=x)
        while var_pred > self.pred_threshold:
            input_machine = self.find_min_distance(input_machine)
            print("passing device_{num}".format(num=input_machine))
            if input_machine in self.device_rf:
                var_pred, y_pred = self.device_rf[input_machine].predict(x=x)
            else:
                print("passing server")
                y_pred = self.rf_precict(rf_list=self.server_rf, x=x)
                return y_pred
        return y_pred




def test():
    data_path = "A:\Research Projects\Distributed_RF\dataset\grinding data.xlsx"
    raw_data = pd.read_excel(data_path)
    raw_y = raw_data.values[:, 0]
    raw_x = raw_data.values[:, 1:]
    train_size = int(len(raw_y) * 0.75)
    data_x = raw_x[:train_size, :]
    data_y = raw_y[:train_size]
    test_x = raw_x[train_size:, :]
    test_y = raw_y[train_size:]

    sim = Partitions(x=data_x, y=data_y, n_devices=5, n_estimators=50,
                     ops=(0, 1), threshold=(0.999, 10), test_interval=5, sampling_factor=5, pred_threshold=0.01)
    #sim.independent_learning()
    sim.shared_learning()
    predict_start = timeit.default_timer()
    #y_pred = sim.global_inferencing(x=test_x)
    y_pred = sim.local_inferencing(x=test_x)
    predict_stop = timeit.default_timer()
    print(np.corrcoef(y_pred, test_y, rowvar=0)[0, 1])
    print('predict time: ', predict_stop - predict_start)

if __name__ == '__main__':
    test()