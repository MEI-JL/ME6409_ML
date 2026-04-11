#!/usr/bin/env python3

# considering google drive space limit, 
# this script will create a copy of original dataset
# but removing all irrelevant files and columns, 
# including those with NaN moments.
import pandas as pd
from pathlib import Path
from helpers.data_management import get_subject_task_paths, _find_suffix_csv_file
from typing import Any, Dict, Tuple

from helpers.constants import *

# change function to keep original structure.
def load_trial_paths(path: Any)-> Tuple[pd.DataFrame, Dict[str,Any]]:
    csv_file_paths = {}
    for prefix in FEATURE_PREFIXES:
         csv_file_path = _find_suffix_csv_file(path, prefix)
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

def reformat_combined_df(df: pd.DataFrame
                         )->Dict[str, pd.DataFrame]:
    # angle
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


def main():
    TASK_PREFIXES = PERIODIC_TASK_PREFIXES + NON_PERIODIC_TASK_PREFIXES
    dataset_folder = "ProcessedData"

    for subject in SUBJECTS:
        for task in TASK_PREFIXES:
            paths = get_subject_task_paths(subject, task, dataset_folder)

            for path in paths:
                df,csv_paths = load_trial_paths(path)
                if df.empty:
                    continue                
                
                dfs = reformat_combined_df(df)

                # write the trimmed csv
                for prefix in FEATURE_PREFIXES:
                    dfs[prefix].to_csv(prefix+"_test.csv", index = False)
                return


if __name__ == '__main__':
	main()