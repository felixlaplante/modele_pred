#####################
# Configuration file

import torch

#####################
# Hyperparameters

cont_date_cols = ['NumDate', 'toy']
cat_date_cols = ['WeekDays', 'BH', 'BH_after', 'Holiday', 'Christmas_break']
weather_cols = ['Temp', 'Temp_s95', 'Temp_s95_min', 'Temp_s95_max', 'Wind', 'Wind_weighted', 'Nebulosity', 'Nebulosity_weighted']
lag_cols = ['Load.1', 'Solar_power.1', 'Wind_power.1']

seq_length = 7
W = 14

lr = 1e-3
batch_size = 128
epochs = 300
hidden_size = 64
num_layers = 2
weight_decay = 0.75
quantile = 0.8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')