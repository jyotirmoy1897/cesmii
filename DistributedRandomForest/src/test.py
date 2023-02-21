import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import sklearn
import timeit
import os
import pickle
from tqdm import tqdm
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt



class Randomforest:
    def __init__(self, data_path, device_name):
        self.data_path = data_path
        self.device_name = device_name

    def confidence_predictor(self, x, mode):
        if mode == "variance":
            result = np.zeros((len(self.RF.estimators_), x.shape[0]))
            count = 0
            for i in self.RF.estimators_:
                y = i.predict(x)
                result[count] = y
                count += 1
            return sum(result.var(axis=0))

    def check_file(self):
        if os.getcwd() != "A:/Research Projects/Distributed_RF":
            os.chdir("A:/Research Projects/Distributed_RF")
        if "RF_parameters" not in os.listdir():
            os.mkdir("RF_parameters")
        if self.device_name + ".pkl" not in os.listdir("RF_parameters"):
            return 0
        return 1

    def save_parameters(self):
        with open("RF_parameters/" + self.device_name + ".pkl", "wb") as fout:
            pickle.dump(self.RF, fout, pickle.HIGHEST_PROTOCOL)

    def read_parameters(self):
        with open("RF_parameters/" + self.device_name + ".pkl", "rb") as fin:
            self.RF = pickle.load(fin)

    def data_preprocess(self, partition_size = 0.75):

        # machine dataset

        raw_data = pd.read_excel(self.data_path)
        raw_data_val = raw_data.values
        np.random.shuffle(raw_data_val)
        raw_y = raw_data_val[:, 0]
        raw_x = raw_data_val[:, 1:]
        train_size = int(len(raw_y) * partition_size)
        train_x = raw_x[:train_size, :]
        train_y = raw_y[:train_size]
        test_x = raw_x[train_size:, :]
        test_y = raw_y[train_size:]

        # appliances energy dataset
        '''
        train_data_path = 'A:\Research Projects\Distributed_RF\dataset\Appliance_training.csv'
        train_data = pd.read_csv(train_data_path)
        test_data_path = 'A:\Research Projects\Distributed_RF\dataset\Appliance_testing.csv'
        test_data = pd.read_csv(test_data_path)
        train_y = train_data.values[:, 0]
        train_x = train_data.values[:, 1:]
        test_y = test_data.values[:, 0]
        test_x = test_data.values[:, 1:]
        '''

        return train_x, train_y, test_x, test_y

    def RF_train(self, x, y, depth=None):
        self.RF = RandomForestRegressor(n_estimators=100, max_depth=depth)
        self.fit_start = timeit.default_timer()
        self.RF.fit(x, y)
        self.fit_stop = timeit.default_timer()
        self.save_parameters()

    def RF_predict(self, x):
        self.predict_start = timeit.default_timer()
        self.y_pred = self.RF.predict(x)
        self.predict_stop = timeit.default_timer()

    def train(self, train_x, train_y, depth=None):
        #if self.check_file():
        if 0:
            self.read_parameters()
            self.fit_start = 0
            self.fit_stop = 0
        else:
            self.RF_train(train_x, train_y, depth=depth)

    def predict(self, test_x):
        self.RF_predict(test_x)

    def eval(self, test_y):
        #print(np.corrcoef(self.y_pred, test_y, rowvar=0)[0, 1])
        #print(mean_absolute_error(self.y_pred, test_y))
        #print('fit time: ', self.fit_stop - self.fit_start)
        #print('predict time: ', self.predict_stop - self.predict_start)
        return np.corrcoef(self.y_pred, test_y, rowvar=0)[0, 1]

def run():
    data_path = "A:/Research Projects/Distributed_RF/dataset/grinding data.xlsx"
    device_name = "cloud"
    DRF = Randomforest(data_path, device_name)

    #cloud
    print('cloud')
    for depth in range(1, 15):
        print('depth: '+str(depth))
        corr_lst = []
        size_lst = []
        for i in range(1):
            train_x, train_y, test_x, test_y = DRF.data_preprocess(partition_size=0.75)
            DRF.train(train_x, train_y, depth=depth)
            DRF.predict(test_x)
            corr_lst.append(DRF.eval(test_y))
            size_lst.append(os.path.getsize('RF_parameters/cloud.pkl'))
        print(np.average(corr_lst))
        print(np.average(size_lst))
        print(train_y)

    '''
    #node
    print('node')
    for depth in range(1, 15):
        print('depth: '+str(depth))
        corr_lst = []
        size_lst = []
        for i in range(50):
            train_x, train_y, test_x, test_y = DRF.data_preprocess(partition_size=0.1875)
            DRF.train(train_x, train_y, depth=depth)
            DRF.predict(test_x)
            corr_lst.append(DRF.eval(test_y))
            size_lst.append(os.path.getsize('RF_parameters/cloud.pkl'))
        print(np.average(corr_lst))
        print(np.average(size_lst))
        '''




    '''
    train_x, train_y, test_x, test_y = DRF.data_preprocess()
    print(test_y)
    DRF.train(train_x, train_y)
    DRF.predict(test_x)
    DRF.eval(test_y)
    para_lst = []
    plt.figure()
    a = sklearn.tree.plot_tree(DRF.RF.estimators_[0])
    print(max([estimator.get_depth() for estimator in DRF.RF.estimators_]))
    print(len(a))
    plt.show()
    
    for i in tqdm(DRF.RF.estimators_):
        para_lst.append(len(sklearn.tree.plot_tree(i)))
    print(np.average(para_lst))
    
    '''

if __name__ == '__main__':
    run()

'''
#appliances energy dataset
train_data_path = 'A:\Research Projects\Distributed_RF\dataset\Appliance_training.csv'
train_data = pd.read_csv(train_data_path)
test_data_path = 'A:\Research Projects\Distributed_RF\dataset\Appliance_testing.csv'
test_data = pd.read_csv(test_data_path)
train_y = train_data.values[:, 0]
train_x = train_data.values[:, 1:]
test_y = test_data.values[:, 0]
test_x = test_data.values[:, 1:]
'''
