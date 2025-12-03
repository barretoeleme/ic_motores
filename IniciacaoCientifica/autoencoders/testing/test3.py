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
import matplotlib.pyplot as plt
    
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
            nn.Linear(128, input_dim),
            nn.Sigmoid() # Sigmoid for [0,1] output
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
MOTOR = "Nabla"
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

target = ['hysteresis', 'joule']

train_dataset = MotorDataset(train_data.drop(columns = target), train_data[target])
test_dataset = MotorDataset(test_data.drop(columns = target), test_data[target])

BATCH_SIZE = 256

train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True)

input_dim = len(train_data.columns.drop(target))
latent_dim = 20

autoencoder_model = Autoencoder(input_dim, latent_dim)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autoencoder_model.to(device)

print(autoencoder_model)

criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder_model.parameters(), lr=0.1)

epochs = 1000
train_losses = []
val_losses = []

for epoch in range(epochs):
    # Training
    autoencoder_model.train()
    running_train_loss = 0.0
    
    for data, _ in train_loader: # _ is the target, which is same as data
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