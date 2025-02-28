import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from strategies.momentum_indicators import calculate_technical_indicators
from utils.loss_functions import CustomMSELoss
from core.ai_models.lstm_model import LSTMModel
from utils.logging_helper import log_message 

DATA_PATH = "data/kospi/kospi_daily_10y.csv"
MODEL_SAVE_PATH = "models/best_lstm_model.pth"
LOOKBACK_DAYS = 240 #🔴test: 60, 120, 240
BATCH_SIZE = 32 # t🔴est: 32, 64
LEARNING_RATE = 0.002 #🔴test: 0.001,0.002, 0.005
NUM_EPOCHS = 50
DROPOUT = 0.6 #🔴test: 0.2, 0.3, 0.4, 0.5 0.6
HIDDEN_SIZE = 48 #🔴test: 64
NUM_LAYERS = 2
weight_decay = 2e-5 #🔴test: 1e-5 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# reduce noise
######################################################################################
def augment_data(df, noise_level=0.02):
    """
    Adds slight Gaussian noise to 'Close' prices to improve generalization.
    noise_level: 2% of the standard deviation of 'Close'
    
    Inject slight noise to improve generalization
    """ 
    noise = np.random.normal(0, 0.02 * df["Close"].std(), size=len(df))
    df["Close"] += noise
    return df

# Load and preprocess data
######################################################################################
def load_data():
    df = pd.read_csv(DATA_PATH)

    df = calculate_technical_indicators(df)  # Apply indicators (RSI, MACD, etc.)
    df = augment_data(df)  # ✅ Apply augmentation before training
    df.dropna(inplace=True)

    features = df[['Open', 'High', 'Low', 'Close', 'Volume'] + list(df.columns[df.columns.str.contains('indicator_')])]
    target = df[['Close']]

    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    target_scaled = scaler.fit_transform(target)

    return features_scaled, target_scaled, scaler

# Custom PyTorch Dataset
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


# Training function
######################################################################################
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler):
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

            # apply loss weighting
            loss_weight = torch.abs(y_batch - y_batch.mean()) + 1  # Prevent div by zero
            loss = (criterion(y_pred, y_batch.view(-1, 1)) * loss_weight).mean()

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)

        # Validation Phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch.view(-1, 1))   
                val_loss += loss.item()

        val_loss /= len(val_loader)  # Compute average validation loss

        # 🔥 Update Learning Rate Scheduler
        scheduler.step(val_loss)  # 🔥 Pass validation loss

        log_message("baseline_lstm", f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        # Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)  # Save best model
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log_message("baseline_lstm", "Early stopping triggered!")
                break

#  Main function to run training and evaluation
######################################################################################
def main():
    log_message("baseline_lstm", "Training started for baseline LSTM model")
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

    weight_decay = 2e-5  
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = CustomMSELoss()

    log_message("baseline_lstm", f"Model initialized. Training for {NUM_EPOCHS} epochs.")

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)

    # Load the best saved model before evaluation
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()

    log_message("baseline_lstm", "Training complete. Running evaluation...")

    # Prepare Test Data for Evaluation
    test_dataset = TimeSeriesDataset(X_val, y_val, LOOKBACK_DAYS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # ✅ Evaluate Model
    predictions, actuals = evaluate_model(model, test_loader, scaler)

    # ✅ Save Predictions for Analysis
    results_df = pd.DataFrame({"Actual": actuals.flatten(), "Predicted": predictions.flatten()})
    results_df.to_csv("predictions_baseline_lstm.csv", index=False)

    log_message("baseline_lstm", "Evaluation complete. Predictions saved to predictions_baseline_lstm.csv.")

#  Evaluate the model on test data
######################################################################################
def evaluate_model(model, test_loader, scaler):
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            y_pred = model(X_batch)
            predictions.append(y_pred.cpu().numpy())
            actuals.append(y_batch.numpy())

    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)

    #inverse transform back to actual stock prices
    predictions_rescaled = scaler.inverse_transform(predictions)
    actuals_rescaled = scaler.inverse_transform(actuals)

    return predictions_rescaled, actuals_rescaled

if __name__ == "__main__":
    main()
