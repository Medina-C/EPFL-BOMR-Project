'''Stores all the constants we need,
   so main.py isn't as cluttered'''

general = {
    'dt': 0.01,
    'goal_buffer': 2,
    'drive_vel': 80,
}

robot = {
    'wheel_rad': 44/2,
    'wheel_base': 95,
    'mtr_val_to_linear': 1/2.5,
    'linear_to_mtr_val': 2.5,
}

ekf = {
    'move_std': 0.0002,
    'measure_std': 0.0001,
}

astolfi = {
    'kp': 0.1,
    'ka': 12,
    'kb': -0.1,
}