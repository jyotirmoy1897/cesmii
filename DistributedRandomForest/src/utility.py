from sklearn.ensemble import RandomForestRegressor
from einops import rearrange
import numpy as np
import torch
import pandas as pd

class rf_reg_preprocess:
    def __init__(self, device, n_estimators):
        self.RF = RandomForestRegressor(n_estimators=n_estimators)
        self.device = device

    def fit(self, train_x, train_y):
        print('generating regression trees...')
        self.RF.fit(train_x, train_y)

    def predict(self, x):
        count = 0
        result = np.zeros((len(self.RF.estimators_), x.shape[0]))
        for i in self.RF.estimators_:
            y_pred = i.predict(x)
            result[count] = y_pred
            count += 1
        return rearrange(torch.tensor(result,
                                      dtype=torch.float32,
                                      device=self.device,
                                      requires_grad=True),
                         'n s-> s n')




def test():
    data_path = "A:\Research Projects\Distributed_RF\dataset\grinding data.xlsx"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    raw_data = pd.read_excel(data_path)
    raw_y = raw_data.values[:, 0]
    raw_x = raw_data.values[:, 1:]
    train_size = int(len(raw_y) * 0.75)
    train_x = raw_x[:train_size, :]
    train_y = raw_y[:train_size]
    test_x = raw_x[train_size:, :]
    test_y = raw_y[train_size:]

    rf = rf_reg_preprocess(device=device, n_estimators=100)
    rf.fit(train_x, train_y)
    train_x = rf.predict(x=train_x)
    test_x = rf.predict(x=test_x)
    train_y = torch.tensor(train_y, dtype=torch.float32, device=device, requires_grad=True)
    test_y = torch.tensor(test_y, dtype=torch.float32, device=device, requires_grad=True)
    print(train_x.shape)
    #print(test_x)


if __name__ == "__main__":
    test()