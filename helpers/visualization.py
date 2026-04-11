import matplotlib.pyplot as plt
import numpy as np

def inspect_knee_moment_dataset(x,y) -> None:

    fig, axes = plt.subplots(3, 2, figsize=(14, 8), sharex='col')
    # knee angle, vel, Thigh acc xyz, thigh gyro xyz, shank acc xyz, shank gyro xyz
    # a column is a separate dataset
    axes[0, 0].plot(x[0,:])
    axes[0, 0].set_title("Knee angle")

    axes[0, 1].plot(x[1,:])
    axes[0, 1].set_title("Knee velocity")

    axes[1, 0].plot(x[2:5].T)
    axes[1, 0].legend(["x", "y", "z"])
    axes[1, 0].set_title("Thigh accel")

    axes[1, 1].plot(x[5:8].T)
    axes[1, 1].legend(["x", "y", "z"])
    axes[1, 1].set_title("Thigh gyro")

    axes[2, 0].plot(x[8:11].T)
    axes[2, 0].legend(["x", "y", "z"])
    axes[2, 0].set_title("Shank accel")

    axes[2, 1].plot(x[11:14].T)
    axes[2, 1].legend(["x", "y", "z"])
    axes[2, 1].set_title("Shank gyro")

    plt.tight_layout()
    plt.show()

    print(y)
    return fig, axes


def plot_loss(train_losses, test_losses):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training & Test Loss Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def inspect_example_data(df):
    fig, axes = plt.subplots(3, 2, figsize=(14, 8), sharex='col')

    for col_idx, (task_type, task_name) in enumerate([
        ('periodic', 'incline_walk_1_down5'),
        ('non_periodic', 'squats_1_0lbs')
    ]):
        trial = df[(df['subject'] == 'AB01') & (df['task'] == task_name)].iloc[:800]

        axes[0, col_idx].plot(trial['RThigh_V_ACCX'].values, label='Thigh Acc X')
        axes[0, col_idx].plot(trial['RShank_V_ACCX'].values, label='Shank Acc X', alpha=0.7)
        axes[0, col_idx].set_ylabel('Accel (m/s\u00b2)')
        axes[0, col_idx].legend(fontsize=8)
        axes[0, col_idx].set_title(f'{task_type.upper()}: {task_name}')

        axes[1, col_idx].plot(trial['knee_angle'].values, color='green')
        axes[1, col_idx].set_ylabel('Knee Angle (deg)')

        axes[2, col_idx].plot(trial['knee_moment'].values, color='red')
        axes[2, col_idx].set_ylabel('Knee Moment (Nm)')
        axes[2, col_idx].set_xlabel('Time Step (200 Hz)')

    plt.tight_layout()
    plt.show()
    print('Notice: periodic tasks show clear repeating patterns, non-periodic tasks do not!')


def prediction_overlay(targets, preds, rmse, r2):
    plt.figure(figsize=(14, 4))
    n_plot = min(300, len(targets))
    time_axis = np.arange(n_plot) / 200  # convert to seconds

    plt.plot(time_axis, targets[:n_plot], label='Actual', alpha=0.8, linewidth=1.5)
    plt.plot(time_axis, preds[:n_plot], label='Predicted', alpha=0.8, linewidth=1.5)
    plt.ylabel('Knee Moment (Nm)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.title(f'CNN Prediction vs Ground Truth  |  RMSE={rmse:.4f} Nm, R\u00b2={r2:.4f}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()