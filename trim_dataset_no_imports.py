#!/usr/bin/env python3

# considering google drive space limit and colab performance,
# this script will create a copy of original dataset
# but removing all irrelevant files and columns, 
# including those with NaN moments.

# this script can be ran without my custom imports.
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Tuple, List

FEATURE_PREFIXES = ["angle", "velocity", "imu_sim", "moment"]
IMU_COLS = [
    'RThigh_V_ACCX','RThigh_V_ACCY','RThigh_V_ACCZ',
    'RThigh_V_GYROX','RThigh_V_GYROY','RThigh_V_GYROZ',
    'RShank_V_ACCX','RShank_V_ACCY','RShank_V_ACCZ',
    'RShank_V_GYROX','RShank_V_GYROY','RShank_V_GYROZ',
]
PERIODIC_TASK_PREFIXES = ["incline_walk", "normal_walk"]
NON_PERIODIC_TASK_PREFIXES = ["sit_to_stand", "squats"]
SUBJECTS = ['AB01','AB02','AB03','AB05','AB06','AB07',
            'AB08','AB09','AB10','AB11','AB12','AB13']


def get_subject_task_paths(subject:str, task_prefix:str, 
                           base:str = "ProcessedData") -> List[Any]:
    base_path = Path(__file__).resolve().parent
    subject_path = base_path / base / subject

    if not subject_path.exists():
        raise ValueError("cannot find subject path/ directory.")
    paths_list = subject_path.glob(task_prefix + "*")

    return sorted(paths_list)


def find_suffix_csv_file(path:Path, suffix:str)->Path|None:
    candidates = path.glob("*" + suffix + ".csv")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


# change function to keep original structure.
def load_trial_paths(path: Any)-> Tuple[pd.DataFrame, Dict[str,Any]]:
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
    df['knee_angle']    = angle['knee_angle_r']
    df['knee_velocity'] = vel['knee_velocity_r']
    for c in IMU_COLS:
        df[c] = imu[c]
    df['knee_moment'] = moment['knee_angle_r_moment']

    return df.dropna(subset=['knee_moment']), csv_file_paths


def reformat_combined_df(df: pd.DataFrame)->Dict[str, pd.DataFrame]:
    dfs = {}
    df_angle = pd.DataFrame()
    df_angle['knee_angle_r'] = df['knee_angle']
    dfs["angle"] = df_angle

    df_velocity = pd.DataFrame()
    df_velocity['knee_velocity_r'] = df['knee_velocity']
    dfs["velocity"] = df_velocity

    df_imu = pd.DataFrame()
    for c in IMU_COLS:
        df_imu[c] = df[c]
    dfs["imu_sim"] = df_imu

    df_moment = pd.DataFrame()
    df_moment['knee_angle_r_moment'] = df['knee_moment']
    dfs["moment"] = df_moment

    return dfs

def replace_dataset_folder_name(path:Path, src_name: str, dst_name: str):
    parts = path.parts
    folder_part_idx = parts.index(src_name)
    folder_children = path.parts[folder_part_idx+1 :]
    folder_parents = path.parts[: folder_part_idx]

    new_parts = folder_parents + (dst_name,) + folder_children # cast to tuple
    
    return Path(*new_parts)

def write_df_to_path(path:Path, df: pd.DataFrame):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(path), index = False)


def main():
    TASK_PREFIXES = PERIODIC_TASK_PREFIXES + NON_PERIODIC_TASK_PREFIXES
    dataset_folder = "ProcessedData"
    trimmed_dataset_folder = "ProcessedDataTrimmed"

    for subject in SUBJECTS:
        for task in TASK_PREFIXES:
            print(subject + " " + task)
            paths = get_subject_task_paths(subject, task, dataset_folder)

            for path in paths:
                df,csv_paths = load_trial_paths(path)
                if df.empty:
                    continue                

                dfs = reformat_combined_df(df)

                # write the trimmed csv
                for prefix in FEATURE_PREFIXES:
                    csv_path = csv_paths[prefix]
                    new_csv_path = replace_dataset_folder_name(
                        csv_path, dataset_folder, trimmed_dataset_folder)
                    write_df_to_path(new_csv_path, dfs[prefix])


if __name__ == '__main__':
	main()