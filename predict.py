#####################
# Set working directory

import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

#####################
# Import libraries

import joblib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

import torch

#####################
# Import custom functions

import config
from prep import load_date_df, load_weather_df, load_fr_co_emissions_df, load_lag_df, load_seq, get_scalers
from fun import PinballLoss, LSTM, predict, rollkalman, agg

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

test_df = pd.read_csv('data/test.csv')
test_df['Date'] = pd.to_datetime(test_df['Date'], utc=True)
test_df['NumDate'] = test_df['Date'].astype('int64')
test_df.set_index('Date', inplace=True)

co_emissions_df = pd.read_csv('data/annual-co-emissions.csv')
co_emissions_df['Year'] = pd.to_datetime(co_emissions_df['Year'].astype(str) + '-07-01', utc=True)
co_emissions_df.set_index('Year', inplace=True)

#####################
# Prepare data

# Train

X_date_df = load_date_df(train_df)
X_weather_df = load_weather_df(train_df)
X_fr_co_emissions_df = load_fr_co_emissions_df(co_emissions_df, train_df)
X_lag_df = load_lag_df(train_df)

X_df = pd.concat([X_date_df, X_weather_df, X_fr_co_emissions_df, X_lag_df], axis=1)
y_df = train_df['Net_demand']

# Test

X_test_date_df = load_date_df(test_df)
X_test_weather_df = load_weather_df(test_df)
X_test_fr_co_emissions_df = load_fr_co_emissions_df(co_emissions_df, test_df).ffill()
X_test_lag_df = load_lag_df(test_df)

X_test_df = pd.concat([X_test_date_df, X_test_weather_df, X_test_fr_co_emissions_df, X_test_lag_df], axis=1)
X_warmup_test_df = pd.concat([X_df[-config.W-config.seq_length+1:], X_test_df], axis=0)
y_warmup_test_df = pd.concat([train_df['Net_demand.1'][-config.W:], test_df['Net_demand.1']], axis=0)

####################
# Scale data

X_scaler = joblib.load('scalers/X_scaler.pkl')
y_scaler = joblib.load('scalers/y_scaler.pkl')

X_scaled = X_scaler.transform(X_df.values.astype(np.float32))
y_scaled = y_scaler.transform(y_df.values.reshape(-1, 1))

X_warmup_test_scaled = X_scaler.transform(X_warmup_test_df.values.astype(np.float32))
y_warmup_test_scaled = y_scaler.transform(y_warmup_test_df.values.reshape(-1, 1))

####################
# Load model

input_dim = X_warmup_test_scaled.shape[1]
output_dim = len(y_scaler.scale_)

model = LSTM(input_dim, output_dim, config.hidden_size, config.num_layers).to(config.device)
model.load_state_dict(torch.load('model/model.pth', weights_only=True))

####################
# Predict

_, pred_train = predict(model, X_scaled)
pred_hidden_test, _ = predict(model, X_warmup_test_scaled)

####################
# Compute quantiles

residuals = y_scaled[config.seq_length-1:] - pred_train
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

pred_mus = np.empty((test_df.shape[0], grid.shape[0]))
pred_sigmas = np.empty((test_df.shape[0], grid.shape[0]))
thetas = []
I = np.eye(theta_0.shape[1])

for i in range(grid.shape[0]):
    alpha, beta = grid[i]
    theta, mus, sigmas = rollkalman(pred_hidden_test, y_warmup_test_scaled, config.W, theta_0, alpha * I, beta * I)
    pred_mus[:, i] = mus
    pred_sigmas[:, i] = sigmas
    thetas.append(theta)

####################
# Aggregate predictions

loss_fn = PinballLoss(config.quantile)

experts = pred_mus + norm.ppf(config.quantile) * pred_sigmas

T, n = experts.shape

eta = torch.sqrt(2 * torch.log(torch.tensor(n)) / T) / B

pred_agg, weights = agg(torch.FloatTensor(experts), torch.FloatTensor(y_warmup_test_scaled[config.W:]), loss_fn, eta)
pred_agg = y_scaler.inverse_transform(pred_agg.detach().numpy().reshape(-1, 1))
weights = weights.detach().numpy()


####################
# Plot

cmap = plt.get_cmap('inferno', n)
colors = cmap(np.arange(n))

fig, ax = plt.subplots(figsize=(12, 6))

cumulative_weights = np.cumsum(weights, axis=1)
bottoms = np.hstack([np.zeros((T, 1)), cumulative_weights[:, :-1]])

for i in range(n):
    ax.fill_between(np.arange(T), bottoms[:, i], cumulative_weights[:, i], color=colors[i], alpha=0.7)

ax.set_xlabel("Time")
ax.set_ylabel("Weight Value")
plt.show()

#####################
# Save

submission = pd.DataFrame({
    'Id': np.arange(1, len(pred_agg) + 1),
    'Net_demand': pred_agg.flatten()
})
submission.to_csv('pred/submission.csv', index=False, sep=',')

print("Submission saved!")