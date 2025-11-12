import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

import datetime
import csv
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

MOTOR = "2D"
PATH = f"../dataset/{MOTOR}/"
TRAIN_FILE = "_all_scaled_train.csv"
TEST_FILE = "_all_scaled_test.csv"

train_data = pd.DataFrame()

train_data = pd.concat([train_data, pd.read_csv(f'{PATH}idiq{TRAIN_FILE}').drop(columns = "Unnamed: 0")], axis = 1)
train_data['speed'] = pd.read_csv(f'{PATH}speed{TRAIN_FILE}')['N']
train_data = pd.concat([train_data, pd.read_csv(f'{PATH}xgeom{TRAIN_FILE}').drop(columns = "Unnamed: 0")], axis = 1)
train_data['hysteresis'] = pd.read_csv(f'{PATH}hysteresis{TRAIN_FILE}')['total']
train_data['joule'] = pd.read_csv(f'{PATH}joule{TRAIN_FILE}')['total']

test_data = pd.DataFrame()

test_data = pd.concat([test_data, pd.read_csv(f'{PATH}idiq{TEST_FILE}').drop(columns = "Unnamed: 0")], axis = 1)
test_data['speed'] = pd.read_csv(f'{PATH}speed{TEST_FILE}')['N']
test_data = pd.concat([test_data, pd.read_csv(f'{PATH}xgeom{TEST_FILE}').drop(columns = "Unnamed: 0")], axis = 1)
test_data['hysteresis'] = pd.read_csv(f'{PATH}hysteresis{TEST_FILE}')['total']
test_data['joule'] = pd.read_csv(f'{PATH}joule{TEST_FILE}')['total']

class RegressionModel(nn.Module):
    
    def __init__(self, input_dim, output_dim, neurons = 5, layers = 1):
        super().__init__()

        modules = []
        
        modules.append(nn.Linear(input_dim, neurons))
        modules.append(nn.ReLU())
        for i in range(layers):
            modules.append(nn.Linear(neurons, neurons))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(neurons, output_dim))
        
        self.linear = nn.Sequential(*modules)
        
    def forward(self, x):
        x = self.linear(x)
        return x
    
class MotorDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

def register_csv(contents, info, MOTOR):
    new_row = pd.DataFrame([contents], columns = info.columns)
    info = pd.concat([info, new_row])
    info.to_csv(f'./data/motor_{MOTOR}_info.csv')
    return info



target = ['hysteresis', 'joule']

train_dataset = MotorDataset(train_data.drop(columns=target), train_data[target])
test_dataset = MotorDataset(test_data.drop(columns=target), test_data[target])

BATCH_SIZE = 64

train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True)

columns = ['neurons', 'layers', 'learn_rate', 'epochs', 'hys_score', 'hys_mse', 'hys_mape', 'jou_score', 'jou_mse', 'jou_mape', 'time']
info = pd.DataFrame(columns = columns)

neurons = np.arange(10, 200, 10)
layers = [1, 2]
learning_rates = [0.1, 0.01]
epochs = 1000

for i in range(len(neurons)):
    for j in range(len(layers)):
        for k in range(len(learning_rates)):
            print(f"\nTraining model --- {neurons[i]}-{layers[j]}-{learning_rates[k]}-{epochs}\n")
            
            input_dim = len(train_data.columns.drop(target))
            output_dim = len(target)
            
            model = RegressionModel(input_dim, output_dim, neurons[i], layers[j])
            
            loss_func = nn.MSELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr = learning_rates[k])
            
            losses = torch.zeros(epochs)

            for a in range(epochs):
                model.train()
                epoch_loss = 0.0
                
                for X_batch, y_batch in train_loader:
                    pred_train = model(X_batch)
                    loss = loss_func(pred_train, y_batch)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        
                    epoch_loss += loss.item()
                    print("batch")

            time = datetime.datetime.now()
            print(f"\tFinished training model at {time}.\n")



            model.eval()

            y_pred_list = []
            y_test_list = []

            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    pred_test = model(X_batch)
                    y_pred_list.append(pred_test)
                    y_test_list.append(y_batch)
                    
            y_pred = torch.cat(y_pred_list)
            y_test = torch.cat(y_test_list)



            hys_score = r2_score(y_test[:, 0], y_pred[:, 0])
            hys_mse = mean_squared_error(y_test[:, 0], y_pred[:, 0])
            hys_mape = mean_absolute_percentage_error(y_test[:, 0], y_pred[:, 0])

            jou_score = r2_score(y_test[:, 1], y_pred[:, 1])
            jou_mse = mean_squared_error(y_test[:, 1], y_pred[:, 1])
            jou_mape = mean_absolute_percentage_error(y_test[:, 1], y_pred[:, 1])

            print(f"\tSpecs:")
            print(f"\t\thys_score: {hys_score}, hys_mse: {hys_mse}, hys_mape: {hys_mape}.\n")
            print(f"\t\tjou_score: {jou_score}, jou_mse: {jou_mse}, jou_mape: {jou_mape}.\n\n")

            contents = [neurons[i], layers[j], learning_rates[k], epochs, hys_score, hys_mse, hys_mape, jou_score, jou_mse, jou_mape, time]
            
            info = register_csv(contents, info, MOTOR)