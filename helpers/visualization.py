import matplotlib.pyplot as plt

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