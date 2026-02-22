# ----------------- Imports ----------------- #

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

# --------------------------------------------------- #

# ----------------- Classes ----------------- #

# our neural network model
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
    
# dataset for our model
class MotorDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
# our autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.ReLU() # The bottleneck layer
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# --------------------------------------------------- #

# ----------------- Functions ----------------- #

# registers results of NN in a csv file
def register_csv(contents, info, MOTOR):
    new_row = pd.DataFrame([contents], columns = info.columns)
    info = pd.concat([info, new_row])
    info.to_csv(f'./data/motor_{MOTOR}_info.csv')
    return info

# --------------------------------------------------- #

# ----------------- Data Loading ----------------- #

MOTOR = "2D"
PATH = f"../../dataset/{MOTOR}/"
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

# --------------------------------------------------- #

# ----------------- Data Preprocessing ----------------- #

target = ['hysteresis', 'joule']

train_dataset = MotorDataset(train_data.drop(columns = target), train_data[target])
test_dataset = MotorDataset(test_data.drop(columns = target), test_data[target])

BATCH_SIZE = 128

train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True)

# --------------------------------------------------- #

input_dim = len(train_data.columns.drop(target))
latent_dim = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

autoencoder_model = Autoencoder(input_dim, latent_dim)
autoencoder_model.to(device)
# print(autoencoder_model)

criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder_model.parameters(), lr=1e-3)
epochs = 50

# --------------------------------------------------- #

# ----------------- Autoencoder Train ----------------- #

train_losses = []
val_losses = []

for epoch in range(epochs):
    # Training
    autoencoder_model.train()
    running_train_loss = 0.0
    
    for data, _ in train_loader:
        data = data.to(device)

        optimizer.zero_grad()
        outputs = autoencoder_model(data)
        loss = criterion(outputs, data)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item() * data.size(0)

    epoch_train_loss = running_train_loss / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)

    # Validation
    autoencoder_model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for data, _ in test_loader: # _ is the target, which is same as data
            data = data.to(device)
            outputs = autoencoder_model(data)
            loss = criterion(outputs, data)
            running_val_loss += loss.item() * data.size(0)

    epoch_val_loss = running_val_loss / len(test_loader.dataset)
    val_losses.append(epoch_val_loss)

    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')

# --------------------------------------------------- #

# ----------------- Neural Network Train ----------------- #

import datetime  # CORREÇÃO NECESSÁRIA

target = ['hysteresis', 'joule']

# ======= CORREÇÃO 1: gerar encoded usando DataFrame ======= #

encoded_train = autoencoder_model.encoder(
    torch.tensor(train_data.drop(columns=target).values, dtype=torch.float32).to(device)
).cpu().detach().numpy()

encoded_test = autoencoder_model.encoder(
    torch.tensor(test_data.drop(columns=target).values, dtype=torch.float32).to(device)
).cpu().detach().numpy()

# ======= CORREÇÃO 2: converter encoded arrays em DataFrame ======= #

new_train_dataset = MotorDataset(pd.DataFrame(encoded_train), train_data[target])
new_test_dataset = MotorDataset(pd.DataFrame(encoded_test), test_data[target])

BATCH_SIZE = 128

new_train_loader = DataLoader(new_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
new_test_loader = DataLoader(new_test_dataset, batch_size=BATCH_SIZE, shuffle=True)

columns = ['neurons', 'layers', 'learn_rate', 'epochs', 'hys_score', 'hys_mse',
           'hys_mape', 'jou_score', 'jou_mse', 'jou_mape', 'time']
info = pd.DataFrame(columns=columns)

neurons = np.arange(10, 200 + 1, 10)
layers = [1, 2, 4]
learning_rates = [0.001, 0.0005, 0.0003]
epochs = 100

for i in range(len(neurons)):
    for j in range(len(layers)):
        for k in range(len(learning_rates)):
            print(f"\nTraining model --- {neurons[i]}-{layers[j]}-{learning_rates[k]}-{epochs}\n")


            input_dim = encoded_train.shape[1]
            output_dim = len(target)

            model = RegressionModel(input_dim, output_dim, neurons[i], layers[j])

            loss_func = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rates[k])

            for a in range(epochs):
                model.train()
                for X, y in new_train_loader:
                    optimizer.zero_grad()
                    pred_train = model(X)
                    loss = loss_func(pred_train, y)
                    loss.backward()
                    optimizer.step()

            time = datetime.datetime.now()
            print(f"\tFinished training model at {time}.\n")

            y_pred_list = []
            y_test_list = []

            model.eval()

            with torch.no_grad():
                for X, y in new_test_loader:
                    pred_test = model(X)
                    y_pred_list.append(pred_test)
                    y_test_list.append(y)

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

            contents = [neurons[i], layers[j], learning_rates[k], epochs,
                        hys_score, hys_mse, hys_mape,
                        jou_score, jou_mse, jou_mape, time]

            info = register_csv(contents, info, MOTOR)