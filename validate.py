#####################
# Set working directory

import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

#####################
# Import libraries

from tqdm import tqdm

from scipy.stats import norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_pinball_loss

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

#####################
# Import custom functions

import config
from prep import load_date_df, load_weather_df, load_fr_co_emissions_df, load_lag_df, load_seq, get_scalers
from fun import train_loop, validate, Load_Xy, LSTM, predict, rollkalman, agg, PinballLoss

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

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.15, shuffle=False)

X_scaler, y_scaler = get_scalers(train_X, train_y)

train_X, train_y = X_scaler.transform(X), y_scaler.transform(y)
val_X, val_y = X_scaler.transform(val_X), y_scaler.transform(val_y)

train_dataset = Load_Xy(*load_seq(train_X, train_y))
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

val_dataset = Load_Xy(*load_seq(val_X, val_y))
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

#####################
# Train model

loss_fn = nn.MSELoss()

input_size = X.shape[1]
output_size = y.shape[1]

model = LSTM(input_size, output_size, config.hidden_size, config.num_layers).to(config.device)

optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

train_losses, val_losses = [], []
for epoch in tqdm(range(config.epochs), desc="Training"):
    t_loss = np.sqrt(train_loop(train_loader, model, loss_fn, optimizer)) * y_scaler.scale_[0]
    v_loss = np.sqrt(validate(val_loader, model, loss_fn)) * y_scaler.scale_[0]
    train_losses.append(t_loss)
    val_losses.append(v_loss)

    if (epoch + 1) % 10 == 0:
        print(f" Epoch {epoch+1}: Train Loss: {t_loss:.4f}, Validation Loss: {v_loss:.4f}")

plt.semilogy(np.arange(len(train_losses)), np.array(train_losses), label='Train Loss')
plt.semilogy(np.arange(len(train_losses)), np.array(val_losses), label='Validation Loss')
plt.title('RMSE Train and validation losses in log scale')
plt.legend()
plt.show()

print(f"Final RMSE Train Loss: {train_losses[-1]:.4f}")
print(f"Final RMSE Validation Loss: {val_losses[-1]:.4f}")

#####################
# Online prediction and validation

X_warmup = X_df[-val_y.shape[0]-config.W-config.seq_length+1:]
y_warmup = train_df['Net_demand.1'][-val_y.shape[0]-config.W:]

####################
# Scale data

X_warmup_scaled = X_scaler.transform(X_warmup.values.astype(np.float32))
y_warmup_scaled = y_scaler.transform(y_warmup.values.reshape(-1, 1))

####################
# Predict

_, pred_train = predict(model, train_X)
pred_hidden_test, _ = predict(model, X_warmup_scaled)

####################
# Compute quantiles

residuals = train_y[config.seq_length-1:] - pred_train
quantile = np.quantile(residuals, config.quantile)
errors = residuals + quantile
B = np.maximum(errors * config.quantile, (1 - config.quantile) * errors).max()

####################
# Predict with online learning

theta_0 = np.hstack([model.fc.bias.detach().cpu().numpy().reshape(1, -1), model.fc.weight.detach().cpu().numpy()])

alphas = np.logspace(-4, 0, 5)
betas = np.logspace(-4, 0, 5)

grid = np.meshgrid(alphas, betas)
grid = np.vstack(grid).T.reshape(-1, 2)

pred_mus = np.empty((val_y.shape[0], grid.shape[0]))
pred_sigmas = np.empty((val_y.shape[0], grid.shape[0]))
thetas = []
I = np.eye(theta_0.shape[1])

for i in range(grid.shape[0]):
    alpha, beta = grid[i]
    theta, mus, sigmas = rollkalman(pred_hidden_test, y_warmup_scaled, config.W, theta_0, alpha * I, beta * I)
    pred_mus[:, i] = mus
    pred_sigmas[:, i] = sigmas
    thetas.append(theta)

####################
# Aggregate predictions

loss_fn = PinballLoss(config.quantile)

experts = pred_mus + norm.ppf(config.quantile) * pred_sigmas

T, n = experts.shape

eta = torch.sqrt(2 * torch.log(torch.tensor(n)) / T) / B

pred_agg, weights = agg(torch.FloatTensor(experts), torch.FloatTensor(y_warmup_scaled[config.W:]), loss_fn, eta)
pred_agg = y_scaler.inverse_transform(pred_agg.detach().numpy().reshape(-1, 1))
weights = weights.detach().numpy()

####################
# Plot prediction 

y_true = y_scaler.inverse_transform(val_y)

plt.plot(y_true, label='True')
plt.plot(pred_agg, label='Predicted')
plt.title('True and predicted values')
plt.legend()
plt.show()

print(f"Pinball loss on the validation set: {mean_pinball_loss(y_true, pred_agg, alpha=config.quantile):.4f}")


