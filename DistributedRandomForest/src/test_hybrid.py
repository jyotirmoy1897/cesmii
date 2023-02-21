import numpy as np
import pandas as pd
from hybrid_reg import RFInitial
from utility import rf_reg_preprocess
import torch
from torch.utils.data import DataLoader
from gen_dataset import GenData
from tqdm import tqdm, trange
from feed_forward import MLP_full
import time
from random import shuffle
from torch import nn
from einops import rearrange
import timeit


def run_epoch(data_loader,
              model,
              optimizer,
              loss_compute,
              sample_num,
              device,
              batch_size,
              weight_decay,
              desc,
              is_train=True):
    running_loss = 0.
    pred_result = []
    ground_truth_result = []
    start = time.time()
    total_batch = sample_num // batch_size + 1

    for i, batch in tqdm(enumerate(data_loader),
                         total=total_batch,
                         desc=desc):
        sample, ground_truth = batch[1], batch[0]
        with torch.set_grad_enabled(is_train):
            out = model(sample)
            out = rearrange(out, 'y n -> (y n)')
            loss = loss_compute(out, ground_truth)
            loss_ = loss
            if is_train:
                '''
                l2_lambda = weight_decay
                for param in model.parameters():
                    if param.requires_grad:
                        loss += l2_lambda * torch.sum(param ** 2)
                '''
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss += loss_.item()
            pred_result += out.tolist()
            ground_truth_result += ground_truth.tolist()

            '''
            print(len(pred_result))
            print(pred_result)
            print("----------------------------------------------------")
            print(len(ground_truth_result))
            print(ground_truth_result)
            '''


    elapsed = time.time() - start
    accuracy = np.corrcoef(pred_result, ground_truth_result, rowvar=0)[0, 1]
    print('\n------ loss: %.3f; accuracy: %.3f; average time: %.4f' %
          (running_loss / total_batch, accuracy, elapsed / sample_num))
    return running_loss / total_batch, accuracy


def runner():
    data_path = "A:\Research Projects\Distributed_RF\dataset\grinding data.xlsx"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    batch_size = 32
    epoch_num = 100
    num_regTrees = 400
    hidden_dim = 512
    output_dim = 1
    num_layers = 3
    weight_decay = 0.0001


    data_path = "A:\Research Projects\Distributed_RF\dataset\grinding data.xlsx"
    raw_data = pd.read_excel(data_path)
    raw_y = raw_data.values[:, 0]
    raw_x = raw_data.values[:, 1:]
    train_size = int(len(raw_y) * 0.75)
    rf_train_x = raw_x[:train_size, :]
    rf_train_y = raw_y[:train_size]
    rf_test_x = raw_x[train_size:, :]
    rf_test_y = raw_y[train_size:]

    '''
    #appliances energy dataset
    train_data_path = 'A:\Research Projects\Distributed_RF\dataset\Appliance_training.csv'
    train_data = pd.read_csv(train_data_path)
    test_data_path = 'A:\Research Projects\Distributed_RF\dataset\Appliance_testing.csv'
    test_data = pd.read_csv(test_data_path)
    rf_train_y = train_data.values[:, 0]
    rf_train_x = train_data.values[:, 1:]
    rf_test_y = test_data.values[:, 0]
    rf_test_x = test_data.values[:, 1:]
    '''


    rf_train_start = timeit.default_timer()
    rf = rf_reg_preprocess(device=device, n_estimators=num_regTrees)
    rf.fit(rf_train_x, rf_train_y)

    train_x = rf.predict(x=rf_train_x)
    train_y = torch.tensor(rf_train_y, dtype=torch.float32, device=device, requires_grad=True)
    rf_train_stop = timeit.default_timer()

    rf_predict_start = timeit.default_timer()
    test_x = rf.predict(x=rf_test_x)
    test_y = torch.tensor(rf_test_y, dtype=torch.float32, device=device, requires_grad=True)
    rf_predict_stop = timeit.default_timer()

    train_start = timeit.default_timer()
    model = MLP_full(input_dim=num_regTrees, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    model = model.to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999),
                                 eps=1e-08, weight_decay=0.0001, amsgrad=False)

    #decay_rate = 0.97
    #lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decay_rate)

    loss_compute = nn.MSELoss().to(device)
    train_stop = timeit.default_timer()
    train_temp = train_stop - train_start + rf_train_stop - rf_train_start
    predict_temp = rf_predict_stop - rf_predict_start


    for epoch in trange(0, epoch_num):
        #y_x = torch.stack(train_x, train_y)
        train_start = timeit.default_timer()
        train_dataloader = DataLoader(GenData(data_x=train_x, data_y=train_y), batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(GenData(data_x=test_x, data_y=test_y), batch_size=batch_size, shuffle=True)
        # Training
        model.train(True)
        loss, accuracy = run_epoch(data_loader=train_dataloader,
                                   model=model,
                                   optimizer=optimizer,
                                   loss_compute=loss_compute,
                                   sample_num=train_y.shape[0],
                                   device=device,
                                   batch_size=batch_size,
                                   weight_decay=weight_decay,
                                   desc="Train Epoch {}".format(epoch + 1),
                                   is_train=True)
        train_stop = timeit.default_timer()
        train_temp += train_stop - train_start
        print('Epoch: {} Evaluating...'.format(epoch + 1))

        # Validation
        predict_start = timeit.default_timer()
        model.eval()
        loss, accuracy = run_epoch(data_loader=test_dataloader,
                                   model=model,
                                   optimizer=optimizer,
                                   loss_compute=loss_compute,
                                   sample_num=test_y.shape[0],
                                   device=device,
                                   batch_size=batch_size,
                                   weight_decay=weight_decay,
                                   desc="Train Epoch {}".format(epoch + 1),
                                   is_train=False)
        predict_stop = timeit.default_timer()
    predict_temp += predict_stop - predict_start
    print(train_temp)
    print(predict_temp)

if __name__ == "__main__":
    runner()


