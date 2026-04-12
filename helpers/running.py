import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from typing import List, Tuple, Dict, Any, Type
from helpers.data_management import (DatasetConfig,
                                     create_LOSO_dataset_dataloader, 
                                     save_checkpoint)
from helpers.constants import *
from helpers.modules import init_model_params

def train_model(
        model: nn.Module, 
        train_loader: DataLoader, 
        test_loader: DataLoader, 
        num_epochs: int = 30, 
        lr: float = 1e-3, 
        device: torch.device = torch.device('cuda')
    ) -> Dict[str,Any]:
    """Train a model and return loss history."""

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer, gamma=0.90)

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
        # update every epoch
        test_losses.append(total_loss / n_samples)
        lr_scheduler.step() 
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print(f'Epoch {epoch:3d} | '
                  f'Train Loss: {train_losses[-1]:.4f} | '
                  f'Test Loss: {test_losses[-1]:.4f}')

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'train_losses': train_losses,
        'test_losses': test_losses
    }
    return checkpoint

def evaluate_model(
        model: nn.Module, 
        data_loader: DataLoader, 
        scaler_y: StandardScaler, 
        device: torch.device = torch.device('cuda')
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Evaluate model and return predictions/targets in original scale.
    In this implementation, scaler is accessed from dataset.scaler_y
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
    preds_orig = scaler_y.inverse_transform(all_preds.reshape(-1, 1)
                                            ).flatten()
    targets_orig = scaler_y.inverse_transform(all_targets.reshape(-1, 1)
                                              ).flatten()

    rmse = np.sqrt(mean_squared_error(targets_orig, preds_orig))
    r2 = r2_score(targets_orig, preds_orig)

    return preds_orig, targets_orig, rmse, r2

def _count_parameters(model:nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def loso_cross_validation(
        subjects: List[str],  
        model_class: Type[nn.Module], # class, not object
        dataset_cfg: DatasetConfig = DatasetConfig(),
        batch_size: int = 32,
        num_epoches: int = 10,
        lr: float = 1e-3,
        device: torch.device = torch.device('cuda'),
        experiment_name: str = "",
        hidden_layer_size: int = HIDDEN_LAYER_SIZE,
    ) -> Tuple[List[float], List[float], nn.Module]:
    ''' main evaluation function'''
    torch.manual_seed(0)
    
    model = model_class(dataset_cfg.ablated_sensors,hidden_layer_size
                        ).to(device)
    rmses = []
    r2s = []
    last_subject = subjects[0]
    checkpoint_name  = experiment_name + model.__class__.__name__
    print("Model "+ checkpoint_name + " parameter size: "
          + str(_count_parameters(model))
          )
    for test_subj in subjects:
        print("Evaluating for test subject " + test_subj)
        # 1. Gather train/test data
        # 2. Fit scaler on TRAINING data only!
        dataset_dataloader = create_LOSO_dataset_dataloader(
                                leave_one_out_subject = test_subj, 
                                dataset_cfg = dataset_cfg,
                                subjects = subjects, 
                                batch_size = batch_size,
                                )
        if dataset_dataloader is None:
            print("skipping...")
            continue
        train_dataset, test_dataset, train_dataloader, test_dataloader \
            = dataset_dataloader # unpack
        print(f'Train: {len(train_dataset)} windows | ',
              f'Test: {len(test_dataset)} windows')
        # 3. Initialize a FRESH model (new weights each fold!)
        init_model_params(model)
        # 4. Train -> Evaluate -> Store RMSE, R² for this fold
        checkpoint = train_model(model, 
                                 train_dataloader, test_dataloader, 
                                 num_epoches, lr, device)
        _, _, rmse, r2 = evaluate_model(model, test_dataloader, 
                                        train_dataset.scaler_y)
        # update storage
        rmses.append(rmse)
        r2s.append(r2)
        last_subject = test_subj

    # just in case, save the last checkpoint
    full_checkpoint_name  = checkpoint_name + "_" + last_subject
    save_checkpoint(checkpoint, full_checkpoint_name)
    print("checkpoint saved in /saved_models/" + full_checkpoint_name)

    print("mean rmse: " + np.mean(rmses))
    print("mean r2: " + np.mean(r2s) )

    return rmses, r2s, model