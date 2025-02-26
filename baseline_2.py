import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from strategies.momentum_indicators import calculate_technical_indicators_2
from utils.loss_functions import CustomMSELoss
from core.ai_models.lstm_model import LSTMModel
from utils.logging_helper import log_message

# ✅ NEW DATASET PATH
DATA_PATH = "data/korean_stock_data.csv"
MODEL_SAVE_PATH = "models/best_lstm_model_2.pth"
LOOKBACK_DAYS = 60
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
DROPOUT = 0.2
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ✅ Load and preprocess data (ADJUSTED FOR NEW COLUMN NAMES)
######################################################################################
def load_data():
    df = pd.read_csv(DATA_PATH)

    # ✅ Convert 'timestamp' to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    # ✅ Apply technical indicators (fix column names)
    df = calculate_technical_indicators_2(df)
    df.dropna(inplace=True)

    # ✅ Adjust feature selection (Fix: Use 'closing_price' instead of 'Close')
    features = df[['opening_price', 'highest_price', 'lowest_price', 'closing_price', 'trading_volume'] +
                  list(df.columns[df.columns.str.contains('indicator_')])]
    target = df[['closing_price']]  # Fix: Use 'closing_price'

    # ✅ Normalize Data
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    target_scaled = scaler.fit_transform(target)

    return features_scaled, target_scaled, scaler

# ✅ Custom PyTorch Dataset
######################################################################################
class TimeSeriesDataset(Dataset):
    def __init__(self, features, targets, lookback_days):
        self.features = features
        self.targets = targets
        self.lookback_days = lookback_days

    def __len__(self):
        return len(self.features) - self.lookback_days

    def __getitem__(self, idx):
        X = self.features[idx:idx + self.lookback_days]
        y = self.targets[idx + self.lookback_days]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# ✅ Training function
######################################################################################
def train_model(model, train_loader, val_loader, criterion, optimizer):
    best_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            y_pred = model(X_batch)

            # ✅ FIX: Ensure y_batch shape matches y_pred
            loss = criterion(y_pred, y_batch.view(-1, 1)) 

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                y_pred = model(X_batch)

                # ✅ FIX: Apply the same fix for validation loss
                loss = criterion(y_pred, y_batch.view(-1, 1)) 

                val_loss += loss.item()

        val_loss /= len(val_loader)
        log_message(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        # Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)  # Save best model
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log_message("Early stopping triggered!")
                break


# ✅ Main function to run training
######################################################################################
def main():
    features_scaled, target_scaled, scaler = load_data()
    
    split_idx = int(len(features_scaled) * 0.8)
    X_train, y_train = features_scaled[:split_idx], target_scaled[:split_idx]
    X_val, y_val = features_scaled[split_idx:], target_scaled[split_idx:]

    train_dataset = TimeSeriesDataset(X_train, y_train, LOOKBACK_DAYS)
    val_dataset = TimeSeriesDataset(X_val, y_val, LOOKBACK_DAYS)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = LSTMModel(
        input_size=X_train.shape[1],
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = CustomMSELoss()

    train_model(model, train_loader, val_loader, criterion, optimizer)
    log_message("Training complete. Model saved.")

if __name__ == "__main__":
    main()
