import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import config
from prep import load_seq

class Load_X(Dataset):
    def __init__(self, X):
        self.X = torch.FloatTensor(X)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]

class Load_Xy(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        return out, self.fc(out)
    

class PinballLoss(nn.Module):
    def __init__(self, quantile):
        super().__init__()
        self.quantile = quantile

    def forward(self, y_pred, y_true):
        errors = y_true - y_pred
        loss = torch.max(self.quantile * errors, (self.quantile - 1) * errors)
        return loss.mean()
    

def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    epoch_loss = 0
    for X, y in dataloader:
        X, y = X.to(config.device), y.to(config.device)
        optimizer.zero_grad()
        _, y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()

    epoch_loss /= len(dataloader)
    return epoch_loss

def validate(dataloader, model, loss_fn):
    model.eval()
    loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(config.device), y.to(config.device)
            _, y_pred = model(X)
            loss += loss_fn(y_pred, y).detach().item()
            
    loss /= len(dataloader)      
    return loss


def predict(model, data):
    X, _ = load_seq(data, None)
    dataloader = DataLoader(Load_X(X), batch_size=config.batch_size, shuffle=False)
    
    model.eval()
    pred_hidden, pred = [], []
    with torch.no_grad():
        for x in dataloader:
            x = x.to(config.device)
            p_hidden, p = model(x)
            p_hidden, p = p_hidden.detach().cpu().numpy(), p.detach().cpu().numpy()
            pred_hidden.append(p_hidden)
            pred.append(p)
            
    pred_hidden, pred = np.vstack(pred_hidden), np.vstack(pred)

    return pred_hidden, pred 


def rollkalman(X_pred, y_true, W, theta_0, V0, Q):
    n, d = X_pred.shape

    X_pred = np.hstack([np.ones((n, 1)), X_pred])
    thetas = np.empty((n - W + 1, d + 1))
    thetas[0] = theta_0.ravel()
    sigmas = np.empty(n - W)
    V = V0

    for i in range(n - W):
        X = X_pred[i : i + W]
        x = X_pred[i + W]
        y = y_true[i + 1 : i + W + 1]
        Vinv = np.linalg.inv(V)
        V = np.linalg.inv(X.T @ X + Vinv)
        theta = V @ (X.T @ y + Vinv @ thetas[i].reshape(-1, 1))
        thetas[i + 1] = theta.ravel()
        sigma2 = np.mean((y - X @ theta)**2)
        sigmas[i] = np.sqrt(sigma2 * (1 + x.T @ V @ x))
        V += Q

    thetas = thetas[1:]
    mus = np.sum(X_pred[config.W:] * thetas, axis=1)

    return thetas[1:], mus, sigmas


def agg(predictions, y, loss_fn, eta):
    T, n = predictions.shape
    log_weights = -torch.full((n,), torch.log(torch.tensor(n, dtype=torch.float32)))
    aggregated = torch.empty(T, dtype=torch.float32)
    weights = torch.empty((T, n), dtype=torch.float32)

    for t in range(T - 1):
        weights[t] = torch.exp(log_weights)
        w = weights[t].clone().detach().requires_grad_()
        agg = torch.dot(w, predictions[t])
        aggregated[t] = agg.detach()
        loss = loss_fn(agg, y[t + 1])
        loss.backward()
        log_weights -= eta * w.grad
        log_weights -= torch.logsumexp(log_weights, dim=0)

    weights[T - 1] = torch.exp(log_weights)
    aggregated[T - 1] = torch.dot(weights[T - 1], predictions[T - 1])

    return aggregated, weights
