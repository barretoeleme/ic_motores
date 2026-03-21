import numpy as np
import pandas as pd
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

# ========================
# CONFIG
# ========================
MOTOR = "V"
PATH = f"../dataset/{MOTOR}/"
TRAIN_FILE = "_all_scaled_train.csv"
TEST_FILE = "_all_scaled_test.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================
# DATA LOADING
# ========================
train_data = pd.DataFrame()

train_data = pd.concat([train_data, pd.read_csv(f'{PATH}idiq{TRAIN_FILE}').drop(columns="Unnamed: 0")], axis=1)
train_data['speed'] = pd.read_csv(f'{PATH}speed{TRAIN_FILE}')['N']
train_data = pd.concat([train_data, pd.read_csv(f'{PATH}xgeom{TRAIN_FILE}').drop(columns="Unnamed: 0")], axis=1)
train_data['hysteresis'] = pd.read_csv(f'{PATH}hysteresis{TRAIN_FILE}')['total']
train_data['joule'] = pd.read_csv(f'{PATH}joule{TRAIN_FILE}')['total']

test_data = pd.DataFrame()

test_data = pd.concat([test_data, pd.read_csv(f'{PATH}idiq{TEST_FILE}').drop(columns="Unnamed: 0")], axis=1)
test_data['speed'] = pd.read_csv(f'{PATH}speed{TEST_FILE}')['N']
test_data = pd.concat([test_data, pd.read_csv(f'{PATH}xgeom{TEST_FILE}').drop(columns="Unnamed: 0")], axis=1)
test_data['hysteresis'] = pd.read_csv(f'{PATH}hysteresis{TEST_FILE}')['total']
test_data['joule'] = pd.read_csv(f'{PATH}joule{TEST_FILE}')['total']

# ========================
# MODEL
# ========================
class RegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim, neurons=5, layers=1):
        super().__init__()

        modules = []
        modules.append(nn.Linear(input_dim, neurons))
        modules.append(nn.ReLU())

        # layers = número total de hidden layers
        for _ in range(layers - 1):
            modules.append(nn.Linear(neurons, neurons))
            modules.append(nn.ReLU())

        modules.append(nn.Linear(neurons, output_dim))

        self.network = nn.Sequential(*modules)

    def forward(self, x):
        return self.network(x)

# ========================
# DATASET
# ========================
class MotorDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

# ========================
# CSV LOGGER
# ========================
def register_csv(contents, info, MOTOR):
    new_row = pd.DataFrame([contents], columns=info.columns)
    info = pd.concat([info, new_row])
    info.to_csv(f'./data/motor_{MOTOR}_info.csv', index=False)
    return info

# ========================
# DATA PREP
# ========================
target = ['hysteresis', 'joule']

train_dataset = MotorDataset(train_data.drop(columns=target), train_data[target])
test_dataset = MotorDataset(test_data.drop(columns=target), test_data[target])

BATCH_SIZE = 256

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)  # ✅ corrigido

# ========================
# EXPERIMENT SETUP
# ========================
columns = ['neurons', 'layers', 'learn_rate', 'epochs',
           'hys_score', 'hys_mse', 'hys_mape',
           'jou_score', 'jou_mse', 'jou_mape', 'time']

info = pd.DataFrame(columns=columns)

neurons_list = np.arange(100, 200 + 1, 10)
layers_list = [1, 2, 4]
learning_rates = [0.001, 0.0005, 0.0003]
epochs = 100

# ========================
# TRAIN LOOP
# ========================
for neurons in neurons_list:
    for layers in layers_list:
        for lr in learning_rates:

            print(f"\nTraining model --- {neurons}-{layers}-{lr}-{epochs}\n")

            input_dim = len(train_data.columns.drop(target))
            output_dim = len(target)

            model = RegressionModel(input_dim, output_dim, neurons, layers).to(device)

            loss_func = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            # ===== TRAIN =====
            for epoch in range(epochs):
                model.train()

                for X, y in train_loader:
                    X, y = X.to(device), y.to(device)

                    optimizer.zero_grad()

                    pred = model(X)
                    loss = loss_func(pred, y)

                    loss.backward()
                    optimizer.step()

            time = datetime.datetime.now()
            print(f"\tFinished training model at {time}.\n")

            # ===== TEST =====
            y_pred_list = []
            y_test_list = []

            model.eval()
            with torch.no_grad():
                for X, y in test_loader:
                    X = X.to(device)

                    pred = model(X)

                    y_pred_list.append(pred.cpu())
                    y_test_list.append(y)

            y_pred = torch.cat(y_pred_list)
            y_test = torch.cat(y_test_list)

            # ===== METRICS =====
            hys_score = r2_score(y_test[:, 0], y_pred[:, 0])
            hys_mse = mean_squared_error(y_test[:, 0], y_pred[:, 0])
            hys_mape = mean_absolute_percentage_error(y_test[:, 0], y_pred[:, 0])

            jou_score = r2_score(y_test[:, 1], y_pred[:, 1])
            jou_mse = mean_squared_error(y_test[:, 1], y_pred[:, 1])
            jou_mape = mean_absolute_percentage_error(y_test[:, 1], y_pred[:, 1])

            print("\tSpecs:")
            print(f"\t\thys_score: {hys_score}, hys_mse: {hys_mse}, hys_mape: {hys_mape}")
            print(f"\t\tjou_score: {jou_score}, jou_mse: {jou_mse}, jou_mape: {jou_mape}\n")

            contents = [neurons, layers, lr, epochs,
                        hys_score, hys_mse, hys_mape,
                        jou_score, jou_mse, jou_mape, time]

            info = register_csv(contents, info, MOTOR)