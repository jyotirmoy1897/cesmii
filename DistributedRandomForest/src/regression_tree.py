'''
With part of code from <<Machine Learning in Action>> by Peter Harrington
'''
import numpy as np
import pandas as pd


def binSplitDataSet(data_x, data_y, feature, value):
    index_left = np.where(data_x[:, feature] > value)
    index_right = np.where(data_x[:, feature] <= value)
    return data_x[index_left], data_y[index_left], data_x[index_right], data_y[index_right]


def regLeaf(data_y):    # returns the regression y value
    return np.mean(data_y)


def regErr(data_y):     # error based on variance
    return np.var(data_y) * data_y.size


def chooseBestSplit(data_x, data_y, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    m, n = data_x.shape
    tolS = ops[0]   # number of samples
    tolN = ops[1]   # error thresholds

    # if all the target variables are the same value: quit and return value
    if np.unique(data_y).size == 1:  # exit cond 1
        return None, leafType(data_y)

    # the choice of the best feature is driven by Reduction in RSS error from mean
    total_error = errType(data_y)
    best_error = np.inf
    best_index = 0
    best_value = 0
    for split_index in range(n):
        for split_value in np.unique(data_x[:, split_index]):
            x_left, y_left, x_right, y_right = binSplitDataSet(data_x, data_y, split_index, split_value)
            if (x_right.shape[0] < tolN) or (x_left.shape[0] < tolN):
                continue

            current_error = errType(y_right) + errType(y_left)

            if current_error < best_error:
                best_index = split_index
                best_value = split_value
                best_error = current_error

    # if the decrease (total_error - best_error) is less than a threshold don't do the split
    if (total_error - best_error) < tolS:
        return None, leafType(data_y)  # exit cond 2

    x_left, y_left, x_right, y_right = binSplitDataSet(data_x, data_y, best_index, best_value)
    if (x_right.shape[0] < tolN) or (x_left.shape[0] < tolN):  # exit cond 3
        return None, leafType(data_y)
    return best_index, best_value  # returns the best feature to split on
    # and the value used for that split


def createTree(data_x, data_y, leafType=regLeaf, errType=regErr,
               ops=(1, 4)):  # assume dataSet is NumPy Mat so we can array filtering
    split_index, split_value = chooseBestSplit(data_x, data_y, leafType, errType, ops)  # choose the best split
    if split_index == None:
        return split_value  # if the splitting hit a stop condition return val
    retTree = {}
    retTree['split_index'] = split_index
    retTree['split_value'] = split_value
    x_left, y_left, x_right, y_right = binSplitDataSet(data_x, data_y, split_index, split_value)
    retTree['left'] = createTree(x_left, y_left, leafType, errType, ops)
    retTree['right'] = createTree(x_right, y_right, leafType, errType, ops)
    return retTree


def isTree(obj):
    return (type(obj).__name__ == 'dict')


def getMean(tree):  # collapse and get the mean value of the tree
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0


def prune(tree, testData_x, testData_y):
    if testData_x.shape[0] == 0:
        return getMean(tree)  # if we have no test data collapse the tree

    if (isTree(tree['right']) or isTree(tree['left'])):  # if the branches are not trees try to prune them
        left_testData_x, left_testData_y, right_testData_x, right_testData_y = binSplitDataSet(testData_x,
                                                                                               testData_y,
                                                                                               tree['feature_index'],
                                                                                               tree['value_index'])

    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], left_testData_x, left_testData_y)

    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], right_testData_x, right_testData_y)

    # if they are now both leafs, see if we can merge them
    if not isTree(tree['left']) and not isTree(tree['right']):
        left_testData_x, left_testData_y, right_testData_x, right_testData_y = binSplitDataSet(testData_x,
                                                                                               testData_y,
                                                                                               tree['feature_index'],
                                                                                               tree['value_index'])
        errorNoMerge = np.sum(np.power(left_testData_y - tree['left'], 2)) + \
                       np.sum(np.power(right_testData_y - tree['right'], 2))
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = np.sum(np.power(testData_y - treeMean, 2))

        if errorMerge < errorNoMerge:
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree


def regTreeEval(model):
    return float(model)


def modelTreeEval(model, data_x):
    x = np.c_[np.ones(data_x.shape[0]), data_x]
    return float(x * model)


def treeForeCast(tree, testData_x, modelEval=regTreeEval):
    if not isTree(tree):
        return modelEval(tree)
    current_pos = tree

    while True:
        if testData_x[current_pos['split_index']] > current_pos['split_value']:
            if isTree(current_pos['left']):
                current_pos = current_pos['left']
            else:
                return modelEval(current_pos['left'])
        else:
            if isTree(current_pos['right']):
                current_pos = current_pos['right']
            else:
                return modelEval(current_pos['right'])


def createForeCast(tree, testData_x, modelEval=regTreeEval):
    m, n = testData_x.shape
    y_pred = np.zeros(m)
    for i in range(m):
        y_pred[i] = treeForeCast(tree, testData_x[i], modelEval)

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

    regTree = createTree(data_x, data_y, ops=(0, 4))
    y_pred = createForeCast(regTree, test_x)

    #print(test_x)
    print(np.corrcoef(y_pred, test_y, rowvar=0)[0, 1])

if __name__ == '__main__':
    test()