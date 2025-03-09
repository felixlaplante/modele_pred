#####################
# Set working directory

import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

#####################
# Import libraries

import joblib
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

#####################
# Import custom functions

import config
from prep import load_date_df, load_weather_df, load_fr_co_emissions_df, load_lag_df, load_seq, get_scalers
from fun import train_loop, Load_Xy, LSTM

#####################
# Set seed

np.random.seed(42)
torch.manual_seed(42)

#####################
# Load data

train_df = pd.read_csv('data/train.csv')
train_df['Date'] = pd.to_datetime(train_df['Date'], utc=True)
train_df['NumDate'] = train_df['Date'].astype('int64')
train_df.set_index('Date', inplace=True)

co_emissions_df = pd.read_csv('data/annual-co-emissions.csv')
co_emissions_df['Year'] = pd.to_datetime(co_emissions_df['Year'].astype(str) + '-07-01', utc=True)
co_emissions_df.set_index('Year', inplace=True)

#####################
# Prepare data

X_date_df = load_date_df(train_df)
X_weather_df = load_weather_df(train_df)
X_fr_co_emissions_df = load_fr_co_emissions_df(co_emissions_df, train_df)
X_lag_df = load_lag_df(train_df)

X_df = pd.concat([X_date_df, X_weather_df, X_fr_co_emissions_df, X_lag_df], axis=1)
y_df = train_df['Net_demand']

X = X_df.values.astype(np.float32)
y = y_df.values.reshape(-1, 1).astype(np.float32)

X_scaler, y_scaler = get_scalers(X, y)

train_X, train_y = X_scaler.transform(X), y_scaler.transform(y)

train_full_dataset = Load_Xy(*load_seq(train_X, train_y))
train_full_loader = DataLoader(train_full_dataset, batch_size=config.batch_size, shuffle=True)

#####################
# Train model

loss_fn = nn.MSELoss()

input_size = X.shape[1]
output_size = y.shape[1]

model = LSTM(input_size, output_size, config.hidden_size, config.num_layers).to(config.device)

optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

train_losses = []
for epoch in tqdm(range(config.epochs), desc="Training"):
    t_loss = np.sqrt(train_loop(train_full_loader, model, loss_fn, optimizer)) * y_scaler.scale_[0]
    train_losses.append(t_loss)

    if (epoch + 1) % 10 == 0:
        print(f" Epoch {epoch+1}: Train Loss: {t_loss:.4f}")

plt.semilogy(np.arange(len(train_losses)), np.array(train_losses), label='Train Loss')
plt.title('Train Loss in log scale')
plt.legend()
plt.show()

print(f"Final RMSE Train Loss: {train_losses[-1]:.4f}")

#####################
# Save model

joblib.dump(X_scaler, 'scalers/X_scaler.pkl')
joblib.dump(y_scaler, 'scalers/y_scaler.pkl')

torch.save(model.state_dict(), 'model/model.pth')
