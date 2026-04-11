import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from typing import List, Tuple, Dict, Any, Literal

from helpers.data_management import create_LOSO_dataset_dataloader, save_checkpoint
from helpers.constants import *
from helpers.modules import init_model_params

# from provided notebook

def train_model(model:nn.Module, train_loader:DataLoader, test_loader:DataLoader, 
                num_epochs:int=30, lr:float=1e-3, 
                device = torch.device('cuda')
                )-> Dict[str,Any]:
    """Train a model and return loss history."""

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.90)

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

        lr_scheduler.step()

        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print(f'Epoch {epoch:3d} | Train Loss: {train_losses[-1]:.4f} | Test Loss: {test_losses[-1]:.4f}')

    # checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'train_losses': train_losses,
        'test_losses': test_losses
    }
    return checkpoint


def evaluate_model(model:nn.Module, data_loader:DataLoader, scaler_y:StandardScaler, 
                   device = torch.device('cuda')
                   )-> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Evaluate model and return predictions/targets in original scale.
    In my altered implementation, scaler is accessed from dataset.scaler_y
    """
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


def loso_cross_validation(subjects:List[str],  
                          tasks:List[str], 
                          model:nn.Module,
                          batch_size = 32,
                          num_epoches:int = 30,
                          lr: float = 1e-3,
                          device = torch.device('cuda'),
                          window_size = WINDOW_SIZE,
                          stride = STRIDE,
                          checkpoint_name: str | None = None,
                          ablated_sensor: Literal["angle", "velocity", "imu_sim"] | None = None
                          ) -> Tuple[List[float], List[float]]:
    if checkpoint_name is None:
        checkpoint_name = model.__class__.__name__

    rmses = []
    r2s = []
    last_subject = subjects[0]
    for test_subj in subjects:
        print("Evaluating model with test subject "+ test_subj)
        # 1. Gather train/test data
        # 2. Fit scaler on TRAINING data only!
        dataset_dataloader = create_LOSO_dataset_dataloader(
                                leave_one_out_subject = test_subj, 
                                tasks = tasks, 
                                subjects = subjects, 
                                batch_size = batch_size, 
                                window_size = window_size, 
                                stride = stride,
                                ablated_sensor = ablated_sensor)
        
        if dataset_dataloader is None:
            print("skipping...")
            continue
        last_subject = test_subj
        train_dataset, test_dataset, train_dataloader, test_dataloader = dataset_dataloader
        
        print("Dataset len: train=" + str(len(train_dataset)) + ", test=" + str(len(test_dataset)))
        # 3. Initialize a FRESH model (new weights each fold!)
        init_model_params(model)
        # 4. Train -> Evaluate -> Store RMSE, R² for this fold
        checkpoint = train_model(model, train_dataloader, test_dataloader, 
                                num_epoches, lr, device)
        _, _, rmse, r2 = evaluate_model(model, test_dataloader, train_dataset.scaler_y)
        rmses.append(rmse)
        r2s.append(r2)

    # just in case, save the last checkpoint
    full_checkpoint_name  = checkpoint_name + "_" + last_subject
    save_checkpoint(checkpoint, full_checkpoint_name)
    print("checkpoint saved in /saved_models/" + full_checkpoint_name)

    return rmses, r2s