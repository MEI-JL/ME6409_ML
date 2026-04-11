SUBJECTS = ['AB01','AB02','AB03','AB05','AB06','AB07',
            'AB08','AB09','AB10','AB11','AB12','AB13']

PERIODIC_TASK_PREFIXES = ["incline_walk", "normal_walk"]
NON_PERIODIC_TASK_PREFIXES = ["sit_to_stand", "squats"]

IMU_COLS = [
    'RThigh_V_ACCX','RThigh_V_ACCY','RThigh_V_ACCZ',
    'RThigh_V_GYROX','RThigh_V_GYROY','RThigh_V_GYROZ',
    'RShank_V_ACCX','RShank_V_ACCY','RShank_V_ACCZ',
    'RShank_V_GYROX','RShank_V_GYROY','RShank_V_GYROZ',
]

WINDOW_SIZE = 50   # 50 samples @ 200 Hz = 250 ms of history
STRIDE = 5         # skip 5 samples between windows (for speed)

FEATURE_COLS = [
    'knee_angle', 'knee_velocity',                                    # Kinematics (2)
    'RThigh_V_ACCX', 'RThigh_V_ACCY', 'RThigh_V_ACCZ',             # Thigh accel (3)
    'RThigh_V_GYROX', 'RThigh_V_GYROY', 'RThigh_V_GYROZ',          # Thigh gyro (3)
    'RShank_V_ACCX', 'RShank_V_ACCY', 'RShank_V_ACCZ',             # Shank accel (3)
    'RShank_V_GYROX', 'RShank_V_GYROY', 'RShank_V_GYROZ',          # Shank gyro (3)
]
TARGET_COL = 'knee_moment'


ANG_COL = 'knee_angle'
VEL_COL = 'knee_velocity'

IMU_THIGH_COLS = [
    'RThigh_V_ACCX','RThigh_V_ACCY','RThigh_V_ACCZ',
    'RThigh_V_GYROX','RThigh_V_GYROY','RThigh_V_GYROZ',
]

IMU_SHANK_COLS = [
    'RShank_V_ACCX','RShank_V_ACCY','RShank_V_ACCZ',
    'RShank_V_GYROX','RShank_V_GYROY','RShank_V_GYROZ',
]

FEATURE_PREFIXES = ["angle", "velocity", "imu_sim", "moment"]