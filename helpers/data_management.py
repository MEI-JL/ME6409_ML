import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from typing import Literal, List, Tuple, Any, Dict
from dataclasses import dataclass, field
from helpers.constants import *

# from provided ME6409_ML_Workshop.ipynb
@dataclass
class DatasetConfig:
    # does not contain subjects, since they will change frequently in LOSO
    tasks: list = field(default_factory=lambda: PERIODIC_TASK_PREFIXES + NON_PERIODIC_TASK_PREFIXES)
    window_size: int = WINDOW_SIZE
    stride: int = STRIDE
    dataset_folder: str = "ProcessedData"
    ablated_sensor: Literal["angle", "velocity", "imu_sim"] | None = None


def get_subject_task_paths(subject:str, task_prefix:str, 
                           base:str = "ProcessedData") -> List[Any]:
    base_path = Path(__file__).resolve().parent.parent
    subject_path = base_path / base / subject

    if not subject_path.exists():
        raise ValueError("cannot find subject path/ directory.")
    paths_list = subject_path.glob(task_prefix + "*")

    return sorted(paths_list)

def find_suffix_csv_file(path, suffix):
    candidates = path.glob("*" + suffix + ".csv")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None

def _load_trial(path:Any, #str or posixpath? 
                 ablated_sensor: Literal[
                     "angle", "velocity", "imu_sim", "imu_thigh", "imu_shank"
                     ] | None = None
               )-> pd.DataFrame:

    csv_file_paths = {}
    for prefix in FEATURE_PREFIXES:
         csv_file_path = find_suffix_csv_file(path, prefix)
         if csv_file_path is None: return pd.DataFrame() # if cannot find file, return empty

         csv_file_paths[prefix] = csv_file_path

    angle  = pd.read_csv(csv_file_paths["angle"])
    vel    = pd.read_csv(csv_file_paths["velocity"])
    imu    = pd.read_csv(csv_file_paths["imu_sim"])
    moment = pd.read_csv(csv_file_paths["moment"])

    df = pd.DataFrame()
    if ablated_sensor != "angle":
        df['knee_angle']    = angle['knee_angle_r']
    if ablated_sensor != "velocity":
        df['knee_velocity'] = vel['knee_velocity_r']
    if ablated_sensor != "imu_sim":
        if ablated_sensor != "imu_thigh":
            for c in IMU_THIGH_COLS:
                df[c] = imu[c]
        if ablated_sensor != "imu_shank":
            for c in IMU_SHANK_COLS:
                df[c] = imu[c]

    df['knee_moment'] = moment['knee_angle_r_moment']

    return df.dropna(subset=['knee_moment'])


def _create_windows(X:np.ndarray, y:np.ndarray, 
                   dataset_samples_indices:List[int] | None = None,
                   window_size = WINDOW_SIZE, stride= STRIDE
                   ) -> np.ndarray:
    """
    Create sliding windows from time series data.
    X: (T, n_features), y: (T,)
    Returns: X_win (n_windows, window_size, n_features), y_win (n_windows,)
    """
    X_windows, y_windows = [], []
    for i in range(0, len(X) - window_size + 1, stride):
        # check if start and end of X are in the same sample; if not then skip.
        if dataset_samples_indices is not None:
            if dataset_samples_indices[i] != dataset_samples_indices[i + window_size-1]:
                continue
        X_windows.append(X[i : i + window_size])
        y_windows.append(y[i + window_size - 1])  # predict at the LAST time step
    return np.array(X_windows), np.array(y_windows)


# only the virtual encoders and distributed IMUs
# Maybe I need to write the filtered data into new files instead...
class KneeMomentDataset(Dataset):
    def __init__(self, subjects: List[str], 
                 cfg: DatasetConfig = DatasetConfig(),
                 scaler_X: StandardScaler | None = None,
                 scaler_y: StandardScaler | None = None,
                 ):
        super().__init__()
        # stored as list of segemented tensors.
        # since we need to get statistics of all dataset, we want to concatenate them 
        # into one array, but later when creating windows, different time series data 
        # might be concatenated in the same window which is undesired.
        tasks = cfg.tasks
        if not isinstance(tasks, list) or not isinstance(tasks[0], str):
            # or will search for all files containing chars in string
            raise TypeError("tasks must be a list of strings.")
        
        self.valid: bool = False # if cannot find data then this is false
        self.feature_cols = []

        sample_index = int(0)
        dataset_samples_indices = []
        dataset_X = [] # list of np arrays, (14,).
        dataset_y = [] # list of np floats

        # resolve path and load from CSV file.
        for subject in subjects:
            for task in tasks:
                paths = get_subject_task_paths(subject, task, cfg.dataset_folder)

                for path in paths:
                    df = _load_trial(path, ablated_sensor=cfg.ablated_sensor)
                    if df.empty:
                        continue
                    # load TODO maybe use sine and cosine for IMU angles...
                    
                    self.feature_cols = df.columns.values[:-1].tolist()
                    demo_data = df.copy()
                    demo_data = demo_data.reset_index(drop=True)
                    
                    # X_raw = demo_data[FEATURE_COLS].values  # shape: (T, 14)
                    X_raw = demo_data[self.feature_cols].values  # shape: (T, 14)
                    y_raw = demo_data[TARGET_COL].values    # shape: (T,)
                    sample_indices = [sample_index] * len(y_raw)
                    sample_index = sample_index + int(1)
                    # concat
                    dataset_samples_indices.extend(sample_indices)
                    dataset_X.extend(X_raw)
                    dataset_y.extend(y_raw)
        
        if not dataset_X: # empty
            self.X = []
            self.y = []
            return
        
        self.valid = True
        
        # change list of np arrays into one np array..
        dataset_X =  np.vstack(dataset_X)  #(points, 14)
        dataset_y = np.vstack(dataset_y)  # (points, 1)

        # normalize large data matrix.
        if scaler_X is None and scaler_y is None:
            self.scaler_X = StandardScaler()
            X_scaled = self.scaler_X.fit_transform(dataset_X)
            self.scaler_y = StandardScaler()
            y_scaled = self.scaler_y.fit_transform(dataset_y.reshape(-1, 1)).flatten()
        else:
            self.scaler_X = scaler_X
            X_scaled = self.scaler_X.transform(dataset_X)
            self.scaler_y = scaler_y
            y_scaled = self.scaler_y.transform(dataset_y.reshape(-1, 1)).flatten()

        # import pdb; pdb.set_trace()

        # split them into windows - this is the data pulled in getitem
        X_windows, y_windows = _create_windows(X_scaled, y_scaled, dataset_samples_indices, 
                                                cfg.window_size, cfg.stride)
        # conv1d input: (N, C_in, L) / (batch, channels, length)
        # lstm input: (L, N, H_in) when batch_first = False (H is input size)
        # but for consistency, always place batches at the first dim.
        # permute in module instead.
        # import pdb; pdb.set_trace()

        self.X = torch.tensor(X_windows, dtype=torch.float32).permute(0, 2, 1) # C, L
        self.y = torch.tensor(y_windows, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_LOSO_dataset_dataloader(leave_one_out_subject: str,
                                   subjects: List[str] = SUBJECTS, # different from that in dataset
                                   dataset_cfg: DatasetConfig = DatasetConfig(),
                                   batch_size = 32,
                                   ) -> Tuple[Dataset, Dataset, DataLoader, DataLoader] | None:
    # returns train/test dataset and dataloader.
    # test (more like validation) set is from the left out subject

    # construct remaining subject list
    train_subjects = [s for s in subjects if s != leave_one_out_subject]

    train_dataset = KneeMomentDataset(train_subjects, cfg = dataset_cfg)
    test_dataset = KneeMomentDataset([leave_one_out_subject], cfg = dataset_cfg,
                                    scaler_X = train_dataset.scaler_X,
                                    scaler_y = train_dataset.scaler_y) # Fit scaler on TRAINING data only!
    
    if not (train_dataset.valid and test_dataset.valid):
        print("Data not found for subject "+ leave_one_out_subject +", returning None")
        return None

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    return train_dataset, test_dataset, train_dataloader, test_dataloader


def save_checkpoint(checkpoint:Dict[str,Any], model_name:str, base_path_name = "/saved_models")->None:
    Path(base_path_name).parent.mkdir(parents=True, exist_ok=True)
    saved_model_path = Path(base_path_name)

    torch.save(checkpoint, "." + str(saved_model_path / (model_name +".pt")))


def load_model(model:torch.nn.Module, model_name:str,base_path_name = "/saved_models")->None:
    # saved_model_path = Path(base_path_name)
    file_path =  "." + str(Path(base_path_name) / (model_name +".pt"))
    checkpoint = torch.load(file_path, weights_only=True)

    model.load_state_dict(checkpoint['model_state_dict'])


def load_checkpoint(model:torch.nn.Module, 
                    optimizer: torch.optim.Adam,
                    lr_scheduler: torch.optim.lr_scheduler.ExponentialLR,
                    model_name:str,base_path_name = "/saved_models"
                    )-> Tuple[int, List[float],List[float]]:
    # saved_model_path = Path(base_path_name)
    file_path =  "." + str(Path(base_path_name) / (model_name +".pt"))

    checkpoint = torch.load(file_path, weights_only=True)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

    return checkpoint['epoch'], checkpoint['train_losses'], checkpoint['test_losses']

