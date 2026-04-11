import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# from provided notebook

def train_model(model, train_loader, test_loader, num_epochs=30, lr=1e-3, 
                device = torch.device('cuda')):
    """Train a model and return loss history."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    train_losses, test_losses = [], []

    for epoch in range(num_epochs):
        # --- Training ---
        model.train()
        total_loss, n_samples = 0.0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)
            n_samples += X_batch.size(0)
        train_losses.append(total_loss / n_samples)

        # --- Evaluation (no gradient computation needed) ---
        model.eval()
        total_loss, n_samples = 0.0, 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                loss = loss_fn(preds, y_batch)
                total_loss += loss.item() * X_batch.size(0)
                n_samples += X_batch.size(0)
        test_losses.append(total_loss / n_samples)

        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print(f'Epoch {epoch:3d} | Train Loss: {train_losses[-1]:.4f} | Test Loss: {test_losses[-1]:.4f}')

    return train_losses, test_losses


def evaluate_model(model, data_loader, scaler_y, device = torch.device('cuda')):
    """Evaluate model and return predictions/targets in original scale."""
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y_batch.numpy())

    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()

    # Inverse transform back to original scale (Nm)
    preds_orig = scaler_y.inverse_transform(all_preds.reshape(-1, 1)).flatten()
    targets_orig = scaler_y.inverse_transform(all_targets.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(targets_orig, preds_orig))
    r2 = r2_score(targets_orig, preds_orig)

    return preds_orig, targets_orig, rmse, r2


def loso_cross_validation(subjects):
    for test_subj in subjects:
        train_subjs = [s for s in subjects if s != test_subj]
        # 1. Gather train/test data
        # 2. Fit scaler on TRAINING data only!
        # 3. Initialize a FRESH model (new weights each fold!)
        # 4. Train -> Evaluate -> Store RMSE, R² for this fold
    pass