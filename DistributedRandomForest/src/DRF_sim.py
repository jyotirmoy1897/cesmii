import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import sklearn
from random import randrange
from tqdm import tqdm
import timeit
import time


class Partitions:
    def __init__(self, data_path, num_devices, threshold, cloud_estimators=100):
        self.data_path = data_path
        self.num_devices = num_devices
        self.pred_threshold = threshold
        self.mode = "variance"
        self.device_data = {}
        self.device_test_data = {}
        self.test_data = None
        self.train_data = None
        self.device_rf = {}
        self.server_rf = None
        self.cloud_estimators = cloud_estimators

    def rf_sampling(self, rf_list):
        n = len(rf_list)
        rf_sampled = []
        for i in range(self.cloud_estimators):
            sample_idx = np.random.choice(n, randrange(1, n + 1), replace=True)
            rf_sampled.append(rf_list[sample_idx[0]])
        return rf_sampled

    def find_min_distance(self, machine):
        return machine+1

    def confidence_predictor(self, rf, x, mode):
        if mode == "variance":
            result = np.zeros((len(rf.estimators_), x.shape[0]))
            count = 0
            for i in rf.estimators_:
                y = i.predict(x)
                result[count] = y
                count += 1
            return sum(result.var(axis=0))

    def data_preprocess(self):

        raw_data = pd.read_excel(self.data_path)
        raw_data_val = raw_data.values
        np.random.shuffle(raw_data_val)

        train_size = int(len(raw_data_val) * 0.75)
        self.train_data = raw_data_val[:train_size]
        self.test_data = raw_data_val[train_size:]
        interval = 0
        test_data_interval = 0
        for i in range(self.num_devices):
            if i == self.num_devices - 1:
                self.device_data[i] = self.train_data[interval:]
                self.device_test_data[i] = self.test_data[test_data_interval:]
            else:
                self.device_data[i] = self.train_data[interval:interval+int(train_size/self.num_devices)]
                self.device_test_data[i] = self.test_data[test_data_interval:test_data_interval+int(len(self.test_data)/self.num_devices)]
            interval += int(train_size / self.num_devices)
            test_data_interval += int(len(self.test_data)/self.num_devices)

    '''
        train_data_path = 'A:\Research Projects\Distributed_RF\dataset\Appliance_training.csv'
        self.train_data = pd.read_csv(train_data_path).values
        np.random.shuffle(self.train_data)
        test_data_path = 'A:\Research Projects\Distributed_RF\dataset\Appliance_testing.csv'
        self.test_data = pd.read_csv(test_data_path).values
        np.random.shuffle(self.test_data)
        train_size = len(self.train_data)
        interval = 0
        for i in range(self.num_devices):
            self.device_data[i] = self.train_data[interval:interval + int(train_size / self.num_devices)]
            interval += int(train_size / self.num_devices)
            '''


    #Data sharing
    def s1_train(self):
        for i in range(self.num_devices):
            x = self.device_data[i][:, 1:]
            y = self.device_data[i][:, 0]
            rf = RandomForestRegressor(n_estimators=100)
            self.device_rf[i] = rf.fit(x, y)
        self.server_rf = RandomForestRegressor(n_estimators=self.cloud_estimators)
        self.server_rf.fit(self.train_data[:, 1:], self.train_data[:, 0])

    #No data sharing
    def s2_train(self):
        rf_list = []
        for i in range(self.num_devices):
            x = self.device_data[i][:, 1:]
            y = self.device_data[i][:, 0]
            rf = RandomForestRegressor(n_estimators=100)
            rf_fit = rf.fit(x, y)
            self.device_rf[i] = rf_fit
            rf_list += rf_fit.estimators_
        self.server_rf = RandomForestRegressor(n_estimators=self.cloud_estimators)
        self.server_rf.fit(self.train_data[:, 1:], self.train_data[:, 0])
        self.server_rf.estimators_ = self.rf_sampling(rf_list)

    #model simplication
    def ms_train(self, depth=None):
        '''
        self.server_rf = RandomForestRegressor(n_estimators=self.cloud_estimators, max_depth=depth)
        self.server_rf.fit(self.train_data[:, 1:], self.train_data[:, 0])
        for i in range(self.num_devices):
            self.device_rf[i] = self.server_rf
        '''
        for i in range(self.num_devices):
            x = self.device_data[i][:, 1:]
            y = self.device_data[i][:, 0]
            rf = RandomForestRegressor(n_estimators=100)
            self.device_rf[i] = rf.fit(x, y)

    #Go server first
    def s1_inf(self):
        input_machine = 0
        device_count = 1
        y_pred = self.device_rf[input_machine].predict(self.test_data[:, 1:])
        var_pred = self.confidence_predictor(self.device_rf[input_machine], self.test_data[:, 1:], self.mode)
        #Go server
        if var_pred > self.pred_threshold:
            device_count += 1
            y_pred = self.server_rf.predict(self.test_data[:, 1:])

        return y_pred, device_count

    # Go nearest device first
    def s2_inf(self):
        input_machine = 0
        device_count = 1
        y_pred = self.device_rf[input_machine].predict(self.test_data[:, 1:])
        var_pred = self.confidence_predictor(self.device_rf[input_machine], self.test_data[:, 1:], self.mode)
        while var_pred > self.pred_threshold:
            input_machine = self.find_min_distance(input_machine)
            device_count += 1
            #Go next device
            if input_machine in self.device_rf:
                y_pred = self.device_rf[input_machine].predict(self.test_data[:, 1:])
                var_pred = self.confidence_predictor(self.device_rf[input_machine], self.test_data[:, 1:], self.mode)
            #Go server
            else:
                y_pred = self.server_rf.predict(self.test_data[:, 1:])
                return y_pred, device_count
        return y_pred, device_count

    def dl_node_inf(self):
        corr_lst = []
        for i in range(self.num_devices):
            y_pred = self.device_rf[i].predict(self.device_test_data[i][:, 1:])
            corr_lst.append(np.corrcoef(y_pred, self.device_test_data[i][:, 0], rowvar=0)[0, 1])
            #corr_lst.append(np.corrcoef(y_pred, self.test_data[:, 0], rowvar=0)[0, 1])

            #print(len(self.device_data[i][:, 0]))
            #print(self.device_test_data[i][:, 0])
        return corr_lst


def test():
    data_path = "A:\Research Projects\Distributed_RF\dataset\grinding data.xlsx"
    #temp: var_threshold = 0.5
    threshold_lst = [0.3 ,0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    devices = [2, 4]

    # limited depth_node_testing
    print('limited depth_node_testing')
    #predict_time_lst = []
    for depth in range(1, 15):
        print('depth: ' + str(depth))
        corr_lst = []
        for i in range(50):
            sim = Partitions(data_path=data_path, num_devices=4, threshold=0.4)
            sim.data_preprocess()
            sim.ms_train(depth=depth)
            #predict_start = timeit.default_timer()
            #y_pred = sim.dl_node_inf()
            #predict_stop = timeit.default_timer()
            #predict_time_lst.append(predict_stop - predict_start)
            corr_lst.append(sim.dl_node_inf())

        #print(np.average(predict_time_lst))
        val = np.average(corr_lst, axis=0)
        for i in val:
            print(i)
        #print(np.average(corr_lst, axis=0))


    '''
    #unlimited depth_i1
    print('unlimited depth_i1')
    predict_time_lst = []
    corr_lst = []
    y_pred_lst = []
    for i in tqdm(range(50)):
        sim = Partitions(data_path=data_path, num_devices=4, threshold=0.4)
        sim.data_preprocess()
        sim.s1_train()
        predict_start = timeit.default_timer()
        y_pred, _ = sim.s1_inf()
        predict_stop = timeit.default_timer()
        y_pred_lst.append(y_pred)
        predict_time_lst.append(predict_stop - predict_start)
        corr_lst.append(np.corrcoef(y_pred, sim.test_data[:, 0], rowvar=0)[0, 1])
    print()
    print(np.average(predict_time_lst))
    print(np.average(corr_lst))

    # unlimited depth_i2
    print('unlimited depth_i2')
    predict_time_lst = []
    corr_lst = []
    y_pred_lst = []
    for i in tqdm(range(50)):
        sim = Partitions(data_path=data_path, num_devices=4, threshold=0.4)
        sim.data_preprocess()
        sim.s1_train()
        predict_start = timeit.default_timer()
        y_pred, _ = sim.s2_inf()
        predict_stop = timeit.default_timer()
        y_pred_lst.append(y_pred)
        predict_time_lst.append(predict_stop - predict_start)
        corr_lst.append(np.corrcoef(y_pred, sim.test_data[:, 0], rowvar=0)[0, 1])
    print()
    print(np.average(predict_time_lst))
    print(np.average(corr_lst))

    #limited depth_i1
    print('limited depth_i1')
    predict_time_lst = []
    for depth in range(1, 15):
        print('depth: '+str(depth))
        corr_lst = []
        for i in range(50):
            sim = Partitions(data_path=data_path, num_devices=4, threshold=0.4)
            sim.data_preprocess()
            sim.ms_train(depth=depth)
            predict_start = timeit.default_timer()
            y_pred, _ = sim.s1_inf()
            predict_stop = timeit.default_timer()
            predict_time_lst.append(predict_stop - predict_start)
            corr_lst.append(np.corrcoef(y_pred, sim.test_data[:, 0], rowvar=0)[0, 1])

        print(np.average(predict_time_lst))
        print(np.average(corr_lst))

    #limited depth_i2
    print('limited depth_i2')
    predict_time_lst = []
    for depth in range(1, 15):
        print('depth: '+str(depth))
        corr_lst = []
        for i in range(50):
            sim = Partitions(data_path=data_path, num_devices=4, threshold=0.4)
            sim.data_preprocess()
            sim.ms_train(depth=depth)
            predict_start = timeit.default_timer()
            y_pred, _ = sim.s2_inf()
            predict_stop = timeit.default_timer()
            predict_time_lst.append(predict_stop - predict_start)
            corr_lst.append(np.corrcoef(y_pred, sim.test_data[:, 0], rowvar=0)[0, 1])
        print(np.average(predict_time_lst))
        print(np.average(corr_lst))
        '''


    '''
    print('s1_i1')
    for threshold in threshold_lst:
        print(threshold)
        for num_devices in devices:
            print(num_devices)
            predict_time_lst = []
            corr_lst = []
            mse_lst = []
            y_pred_lst = []
            for i in tqdm(range(50)):
                sim = Partitions(data_path=data_path, num_devices=num_devices, threshold=threshold)
                sim.data_preprocess()
                sim.s1_train()
                predict_start = timeit.default_timer()
                y_pred = sim.s1_inf()
                predict_stop = timeit.default_timer()
                y_pred_lst.append(y_pred)
                predict_time_lst.append(predict_stop - predict_start)
                corr_lst.append(np.corrcoef(y_pred, sim.test_data[:, 0], rowvar=0)[0, 1])
                mse_lst.append(mean_absolute_error(y_pred, sim.test_data[:, 0]))
            print()
            print(np.average(predict_time_lst))
            print(np.var(predict_time_lst))
            print(np.average(corr_lst))
            print(np.var(corr_lst))
            print(np.average(mse_lst))
            print(np.var(mse_lst))
            '''
    '''
    print('s2_i1')
    for threshold in threshold_lst:
        print(threshold)
        for num_devices in devices:
            print(num_devices)
            predict_time_lst = []
            corr_lst = []
            mse_lst = []
            y_pred_lst = []
            device_lst = []
            for i in tqdm(range(50)):
                sim = Partitions(data_path=data_path, num_devices=num_devices, threshold=threshold)
                sim.data_preprocess()
                sim.s2_train()
                predict_start = timeit.default_timer()
                y_pred, device_count = sim.s1_inf()
                predict_stop = timeit.default_timer()
                y_pred_lst.append(y_pred)
                predict_time_lst.append(predict_stop - predict_start)
                corr_lst.append(np.corrcoef(y_pred, sim.test_data[:, 0], rowvar=0)[0, 1])
                mse_lst.append(mean_absolute_error(y_pred, sim.test_data[:, 0]))
                device_lst.append(device_count)
            print()
            print(np.average(device_lst))
            
            print(np.average(predict_time_lst))
            print(np.var(predict_time_lst))
            print(np.average(corr_lst))
            print(np.var(corr_lst))
            print(np.average(mse_lst))
            print(np.var(mse_lst))
            '''
    '''
    print('s1_i2')
    for threshold in threshold_lst:
        print(threshold)
        for num_devices in devices:
            print(num_devices)
            predict_time_lst = []
            corr_lst = []
            mse_lst = []
            y_pred_lst = []
            for i in tqdm(range(50)):
                sim = Partitions(data_path=data_path, num_devices=num_devices, threshold=threshold)
                sim.data_preprocess()
                sim.s1_train()
                predict_start = timeit.default_timer()
                y_pred = sim.s2_inf()
                predict_stop = timeit.default_timer()
                y_pred_lst.append(y_pred)
                predict_time_lst.append(predict_stop - predict_start)
                corr_lst.append(np.corrcoef(y_pred, sim.test_data[:, 0], rowvar=0)[0, 1])
                mse_lst.append(mean_absolute_error(y_pred, sim.test_data[:, 0]))
            print()
            print(np.average(predict_time_lst))
            print(np.var(predict_time_lst))
            print(np.average(corr_lst))
            print(np.var(corr_lst))
            print(np.average(mse_lst))
            print(np.var(mse_lst))
            '''
    '''
    print('s2_i2')
    for threshold in threshold_lst:
        print(threshold)
        for num_devices in devices:
            print(num_devices)
            predict_time_lst = []
            corr_lst = []
            mse_lst = []
            y_pred_lst = []
            device_lst = []
            for i in tqdm(range(50)):
                sim = Partitions(data_path=data_path, num_devices=num_devices, threshold=threshold)
                sim.data_preprocess()
                sim.s2_train()
                predict_start = timeit.default_timer()
                y_pred, device_count = sim.s2_inf()
                predict_stop = timeit.default_timer()
                y_pred_lst.append(y_pred)
                predict_time_lst.append(predict_stop - predict_start)
                corr_lst.append(np.corrcoef(y_pred, sim.test_data[:, 0], rowvar=0)[0, 1])
                mse_lst.append(mean_absolute_error(y_pred, sim.test_data[:, 0]))
                device_lst.append(device_count)
            print()
            print(np.average(device_lst))
            
            print(np.average(predict_time_lst))
            print(np.var(predict_time_lst))
            print(np.average(corr_lst))
            print(np.var(corr_lst))
            print(np.average(mse_lst))
            print(np.var(mse_lst))
            '''
if __name__ == '__main__':
    test()