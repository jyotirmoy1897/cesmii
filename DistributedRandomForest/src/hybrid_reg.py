import numpy as np
from random import randrange
from regression_tree import createTree
from tqdm import tqdm
from regression_tree import createForeCast
import torch
from einops import rearrange


class RFInitial:
    def __init__(self, n_estimators, ops, threshold, test_interval, device):
        self.n_estimators = n_estimators
        self.ops = ops
        self.reg_trees = None
        self.threshold = threshold
        self.test_interval = test_interval
        self.device = device

    def get_bootstrap_data(self, x, y):
        n = x.shape[0]
        y_x = np.column_stack((y, x))
        np.random.shuffle(y_x)
        datasets = []
        for i in range(self.n_estimators):
            sample_idx = np.random.choice(n, randrange(1, n+1), replace=True)
            bootstrap_y_x = y_x[sample_idx, :]
            bootstrap_x = bootstrap_y_x[:, 1:]
            bootstrap_y = bootstrap_y_x[:, 0]
            datasets.append((bootstrap_x, bootstrap_y))
        return datasets

    def fit(self, x, y):
        self.reg_trees = []
        data = self.get_bootstrap_data(x, y)
        print('fitting...')
        for i in tqdm(range(self.n_estimators)):
            data_x, data_y = data[i]
            if i != 0 and i % self.test_interval == 0:
                if data_x.shape[0] >= self.threshold[1]:
                    test_val = np.corrcoef(self.predict(data_x)[1], data_y, rowvar=0)[0, 1]
                    #print(test_val)
                    if test_val >= self.threshold[0]:
                        break
            self.reg_trees.append(createTree(data_x, data_y, ops=self.ops))

    def predict(self, x):
        sum_pred = np.zeros(x.shape[0])
        sum_var = 0
        count = 0
        result = np.zeros((len(self.reg_trees), x.shape[0]))
        #result = torch.zeros((len(self.reg_trees), x.shape[0]), device=self.device, requires_grad=True)
        for reg_tree in self.reg_trees:
            y_pred = createForeCast(reg_tree, x)
            sum_pred += y_pred
            sum_var += np.var(y_pred)
            result[count] = y_pred
            count += 1
        return sum_var / count, sum_pred / count, rearrange(torch.tensor(result,
                                                                         dtype=torch.float32,
                                                                         device=self.device,
                                                                         requires_grad=True),
                                                            'n s-> s n')

