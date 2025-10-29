import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import datetime
import csv
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

# data loading

train_data = pd.DataFrame()

train_data['hysteresis'] = pd.read_csv('../dataset/2D/hysteresis_all_scaled_train.csv')['total']
train_data['id'] = pd.read_csv('../dataset/2D/idiq_all_scaled_train.csv')['id']
train_data['iq'] = pd.read_csv('../dataset/2D/idiq_all_scaled_train.csv')['iq']
train_data['joule'] = pd.read_csv('../dataset/2D/joule_all_scaled_train.csv')['total']
train_data['speed'] = pd.read_csv('../dataset/2D/speed_all_scaled_train.csv')['N']
train_data['d1'] = pd.read_csv('../dataset/2D/xgeom_all_scaled_train.csv')['d1']
train_data['d2'] = pd.read_csv('../dataset/2D/xgeom_all_scaled_train.csv')['d2']
train_data['d3'] = pd.read_csv('../dataset/2D/xgeom_all_scaled_train.csv')['d3']
train_data['d4'] = pd.read_csv('../dataset/2D/xgeom_all_scaled_train.csv')['d4']
train_data['d5'] = pd.read_csv('../dataset/2D/xgeom_all_scaled_train.csv')['d5']
train_data['d6'] = pd.read_csv('../dataset/2D/xgeom_all_scaled_train.csv')['d6']
train_data['d7'] = pd.read_csv('../dataset/2D/xgeom_all_scaled_train.csv')['d7']
train_data['d8'] = pd.read_csv('../dataset/2D/xgeom_all_scaled_train.csv')['d8']
train_data['d9'] = pd.read_csv('../dataset/2D/xgeom_all_scaled_train.csv')['d9']
train_data['r1'] = pd.read_csv('../dataset/2D/xgeom_all_scaled_train.csv')['r1']
train_data['t1'] = pd.read_csv('../dataset/2D/xgeom_all_scaled_train.csv')['t1']

test_data = pd.DataFrame()

test_data['hysteresis'] = pd.read_csv('../dataset/2D/hysteresis_all_scaled_test.csv')['total']
test_data['id'] = pd.read_csv('../dataset/2D/idiq_all_scaled_test.csv')['id']
test_data['iq'] = pd.read_csv('../dataset/2D/idiq_all_scaled_test.csv')['iq']
test_data['joule'] = pd.read_csv('../dataset/2D/joule_all_scaled_test.csv')['total']
test_data['speed'] = pd.read_csv('../dataset/2D/speed_all_scaled_test.csv')['N']
test_data['d1'] = pd.read_csv('../dataset/2D/xgeom_all_scaled_test.csv')['d1']
test_data['d2'] = pd.read_csv('../dataset/2D/xgeom_all_scaled_test.csv')['d2']
test_data['d3'] = pd.read_csv('../dataset/2D/xgeom_all_scaled_test.csv')['d3']
test_data['d4'] = pd.read_csv('../dataset/2D/xgeom_all_scaled_test.csv')['d4']
test_data['d5'] = pd.read_csv('../dataset/2D/xgeom_all_scaled_test.csv')['d5']
test_data['d6'] = pd.read_csv('../dataset/2D/xgeom_all_scaled_test.csv')['d6']
test_data['d7'] = pd.read_csv('../dataset/2D/xgeom_all_scaled_test.csv')['d7']
test_data['d8'] = pd.read_csv('../dataset/2D/xgeom_all_scaled_test.csv')['d8']
test_data['d9'] = pd.read_csv('../dataset/2D/xgeom_all_scaled_test.csv')['d9']
test_data['r1'] = pd.read_csv('../dataset/2D/xgeom_all_scaled_test.csv')['r1']
test_data['t1'] = pd.read_csv('../dataset/2D/xgeom_all_scaled_test.csv')['t1']



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

def register_csv(contents, info):
    new_row = pd.DataFrame([contents], columns = data.columns)
    info = pd.concat([info, new_row])
    data.to_csv('./data/motor_2D_info.csv')
    return data

def register_txt(contents, info):
    new_row = pd.DataFrame([contents], columns = info.columns)
    
    with open('motor_2D_log.txt'):
        file.write("\n")
        
        file.write(f"Test ID: {new_now.neurons}-{new_row.layers}-{new_row.learn_rate}-{new_row.epochs}\n")
        file.write(f"Test run at {new_row.time}\n")
    
        file.write("\n")
        
        file.write("\t> Parameters:\n")
        file.write(f"\t\t>> Number of neurons: {new_row.neurons}\n")
        file.write(f"\t\t>> Number of layers: {new_row.layers}\n")
        file.write(f"\t\t>> Learning rate: {new_row.learn_rate}\n")
        file.write(f"\t\t>> Number of epochs: {new_row.epochs}\n")
    
        file.write("\n")
    
        file.write("\t> Results:\n")
        file.write(f"\t\t>> Score: {new_row.score}\n")
        file.write(f"\t\t>> Mean squared error: {new_row.mse}\n")
        file.write(f"\t\t>> MAPE: {new_row.mape}\n")
    
        file.write("\n")

target = ['hysteresis', 'joule']

neurons = [5, 10, 15, 20, 25, 30]
layers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
learning_rates = [0.9, 0.5, 0.1, 0.05, 0.01]
epochs = 500

X_train = torch.tensor(train_data.drop(columns = target).values, dtype=torch.float32)
y_train = torch.tensor(train_data[target].values, dtype=torch.float32)
X_test = torch.tensor(test_data.drop(columns = target).values, dtype=torch.float32)
y_test = torch.tensor(test_data[target].values, dtype=torch.float32)

columns = ['neurons', 'layers', 'learn_rate', 'epochs', 'score', 'mse', 'mape', 'time']
info = pd.DataFrame(columns = columns)

for i in range(len(neurons)):
    for j in range(len(layers)):
        for k in range(len(learning_rates)):
            print(f"\nTraining model {i+j+k} --- {neurons[i]}-{layers[j]}-{learning_rates[k]}-{epochs}\n")
            input_dim = len(train_data.columns.drop(target))
            output_dim = len(target)
            
            model = RegressionModel(input_dim, output_dim, neurons[i], layers[j])
            
            loss_func = nn.MSELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr = learning_rates[k])
            
            losses = torch.zeros(epochs)

            for a in range(epochs):
                pred = model(X_train)
            
                loss = loss_func(pred, y_train)
                losses[a] = loss
            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            y_pred = model(X_test)
            time = datetime.datetime.now()

            print(f"\tFinished training model at {time}.\n")

            score = r2_score(y_pred[:, 0].detach().numpy(), y_test[:, 0].detach().numpy())
            mse = mean_squared_error(y_pred[:, 0].detach().numpy(), y_test[:, 0].detach().numpy())
            mape = mean_absolute_percentage_error(y_pred[:, 0].detach().numpy(), y_test[:, 0].detach().numpy())

            print(f"\tSpecs: score: {score}, mse: {mse}, mape: {mape}.\n\n")

            contents = [neurons[i], layers[j], learning_rates[k], epochs, score, mse, mape, time]
            
            info = register_csv(contents, info)
            register_txt(contents)