import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from typing import Literal

from helpers.constants import *

# from provided ME6409_ML_Workshop.ipynb

def get_subject_task_paths(subject, task_prefix, base = "ProcessedData"):
    base_path = Path(__file__).resolve().parent.parent
    subject_path = base_path / base / subject

    if not subject_path.exists():
        raise ValueError("cannot find subject pathectory.")
    paths_list = subject_path.glob(task_prefix + "*")

    return sorted(paths_list)

def find_suffix_csv_file(path, suffix):
    candidates = path.glob("*" + suffix + ".csv")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None

def load_trial(path, 
               ablated_sensor: Literal["angle", "velocity", "imu_sim"] | None = None):
    
    angle  = pd.read_csv(find_suffix_csv_file(path, "angle"))
    vel    = pd.read_csv(find_suffix_csv_file(path, "velocity"))
    imu    = pd.read_csv(find_suffix_csv_file(path, "imu_sim"))
    moment = pd.read_csv(find_suffix_csv_file(path, "moment"))

    df = pd.DataFrame()
    if ablated_sensor != "angle":
        df['knee_angle']    = angle['knee_angle_r']
    if ablated_sensor != "velocity":
        df['knee_velocity'] = vel['knee_velocity_r']
    if ablated_sensor != "imu_sim":
        for c in IMU_COLS:
            df[c] = imu[c]
        df['knee_moment'] = moment['knee_angle_r_moment']

    return df.dropna(subset=['knee_moment'])

def create_windows(X, y, dataset_samples_indices = None,
                   window_size = WINDOW_SIZE, stride= STRIDE):
    """
    Create sliding windows from time series data.
    X: (T, n_features), y: (T,)
    Returns: X_win (n_windows, window_size, n_features), y_win (n_windows,)
    """
    X_windows, y_windows = [], []
    for i in range(0, len(X) - window_size + 1, stride):
        # check if start and end of X are in the same sample; if not then skip.
        if dataset_samples_indices is not None:
            if dataset_samples_indices[i] != dataset_samples_indices[i + window_size]:
                continue
        X_windows.append(X[i : i + window_size])
        y_windows.append(y[i + window_size - 1])  # predict at the LAST time step
    return np.array(X_windows), np.array(y_windows)


# only the virtual encoders and distributed IMUs
# Maybe I need to write the filtered data into new files instead...
class KneeMomentDataset(Dataset):
    def __init__(self, subjects, tasks, 
                 window_size = WINDOW_SIZE, 
                 stride = STRIDE):
        super().__init__()
        # stored as list of segemented tensors.
        # since we need to get statistics of all dataset, we want to concatenate them 
        # into one array, but later when creating windows, different time series data 
        # might be concatenated in the same window which is undesired.
        sample_index = int(0)
        dataset_samples_indices = []
        dataset_X = []
        dataset_y = []

        # resolve path and load from CSV file.
        for subject in subjects:
            for task in tasks:
                paths = get_subject_task_paths(subject, task)

                for path in paths:
                    df = load_trial(path)
                    if df.empty:
                        continue
                    # load
                    demo_data = df.copy()
                    demo_data = demo_data.reset_index(drop=True)
                    X_raw = demo_data[FEATURE_COLS].values  # shape: (T, 14)
                    y_raw = demo_data[TARGET_COL].values    # shape: (T,)
                    sample_indices = [sample_index] * len(y_raw)
                    sample_index = sample_index + int(1)
                    # concat
                    dataset_samples_indices.extend(sample_indices)
                    dataset_X.extend(X_raw)
                    dataset_y.extend(y_raw)

        # normalize large data matrix. TODO see if need to cast into np format.
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X_raw)

        scaler_y = StandardScaler()
        y_scaled = scaler_y.fit_transform(y_raw.reshape(-1, 1)).flatten()

        # split them into windows - this is the data pulled in getitem
        X_windows, y_windows = create_windows(X_scaled, y_scaled, dataset_samples_indices, 
                                                window_size, stride)
        
        # conv1d input: (N, C_in, L) / (batch, channels, length)
        # lstm input: (L, N, H_in) when batch_first = False (H is input size)
        # but for consistency, always place batches at the first dim.
        # permute in module instead.
        self.X = torch.tensor(X_windows, dtype=torch.float32).permute(0, 2, 1)
        self.y = torch.tensor(y_windows, dtype=torch.float32).unsqueeze(-1)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]


def create_LOSO_dataset_dataloader(leave_one_out_subject, tasks, 
                              window_size = WINDOW_SIZE,
                              stride = STRIDE):
    # returns train/test dataset and dataloader.
    # test (more like validation) set is from the left out subject

    # construct remaining subject list
    train_subjects = []
    for subject in SUBJECTS:
        if subject != leave_one_out_subject:
            train_subjects.append(subject)

    train_dataset = KneeMomentDataset(train_subjects,tasks, window_size, stride)
    test_dataset = KneeMomentDataset([leave_one_out_subject], tasks, window_size, stride)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

    return train_loader, test_loader, train_dataset, test_dataset