import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from pathlib import Path

# =========================
# AUTOENCODER FLEXÍVEL
# =========================
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=20, neurons=128, layers=1):
        super().__init__()

        encoder_layers = [nn.Linear(input_dim, neurons), nn.ReLU()]

        for _ in range(layers):
            encoder_layers.append(nn.Linear(neurons, neurons))
            encoder_layers.append(nn.ReLU())

        encoder_layers.append(nn.Linear(neurons, latent_dim))

        decoder_layers = [nn.Linear(latent_dim, neurons), nn.ReLU()]

        for _ in range(layers):
            decoder_layers.append(nn.Linear(neurons, neurons))
            decoder_layers.append(nn.ReLU())

        decoder_layers.append(nn.Linear(neurons, input_dim))

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))


# =========================
# LOAD DATA
# =========================
def get_data(motor):
    base_path = Path(__file__).resolve().parent.parent.parent / "dataset" / motor

    TRAIN_FILE = "_all_scaled_train.csv"
    TEST_FILE = "_all_scaled_test.csv"

    def build(file):
        df = pd.DataFrame()

        df = pd.concat([
            df,
            pd.read_csv(base_path / f"idiq{file}").drop(columns="Unnamed: 0")
        ], axis=1)

        df["speed"] = pd.read_csv(base_path / f"speed{file}")["N"]

        df = pd.concat([
            df,
            pd.read_csv(base_path / f"xgeom{file}").drop(columns="Unnamed: 0")
        ], axis=1)

        df["hysteresis"] = pd.read_csv(base_path / f"hysteresis{file}")["total"]
        df["joule"] = pd.read_csv(base_path / f"joule{file}")["total"]

        return df

    return build(TRAIN_FILE), build(TEST_FILE)


# =========================
# TREINO AUTOENCODER
# =========================
def train_autoencoder(model, X_train, X_eval, X_test, lr=1e-3, epochs=50):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_train = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    X_eval  = torch.tensor(X_eval.values, dtype=torch.float32).to(device)
    X_test  = torch.tensor(X_test.values, dtype=torch.float32).to(device)

    for epoch in range(epochs):
        model.train()

        optimizer.zero_grad()
        loss = criterion(model(X_train), X_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_loss = criterion(model(X_train), X_train).item()
        eval_loss  = criterion(model(X_eval), X_eval).item()
        test_loss  = criterion(model(X_test), X_test).item()

    return train_loss, eval_loss, test_loss


# =========================
# MAIN
# =========================
motors = ["2D", "Nabla", "V"]

neurons_list = np.arange(100, 201, 10)
layers_list = [1, 2, 4]
lrs = [0.001, 0.0005, 0.0003]

for motor in motors:

    print(f"\n=== MOTOR {motor} ===")

    # =========================
    # LOAD + SPLIT
    # =========================
    train_data, test_data = get_data(motor)

    full = pd.concat([train_data, test_data]).reset_index(drop=True)

    train_df, temp_df = train_test_split(full, test_size=0.30, random_state=42)
    eval_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    target = ['hysteresis', 'joule']

    X_train = train_df.drop(columns=target)
    X_eval  = eval_df.drop(columns=target)
    X_test  = test_df.drop(columns=target)

    input_dim = X_train.shape[1]

    # =========================
    # GRID SEARCH
    # =========================
    results = []

    for neurons in neurons_list:
        for layers in layers_list:
            for lr in lrs:

                print(f"AE: {neurons} neurons | {layers} layers | lr={lr}")

                model = Autoencoder(
                    input_dim=input_dim,
                    neurons=neurons,
                    layers=layers
                )

                train_loss, eval_loss, test_loss = train_autoencoder(
                    model,
                    X_train,
                    X_eval,
                    X_test,
                    lr=lr,
                    epochs=50
                )

                results.append([
                    neurons, layers, lr,
                    train_loss, eval_loss, test_loss
                ])

    # =========================
    # SAVE RESULTS
    # =========================
    df_results = pd.DataFrame(results, columns=[
        "neurons", "layers", "lr",
        "train_loss", "eval_loss", "test_loss"
    ])

    results_path = Path(__file__).resolve().parent.parent / "results"
    results_path.mkdir(exist_ok=True)

    df_results.to_csv(results_path / f"{motor}_autoencoder_grid.csv", index=False)

    print(f"Saved results for {motor}")