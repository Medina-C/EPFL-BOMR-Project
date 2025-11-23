'''Stores all the constants we need,
   so main.py isn't as cluttered'''

general = {
    'dt': 0.12,
    'goal_buffer': 50,
    'drive_vel': 130,
    'max_vel': 130,
    'max_accel': 20000,
    'max_w': 0.7,
}

robot = {
    'wheel_rad': 44/2,
    'wheel_base': 95,
    'mtr_val_to_linear': 1/2.5,
    'linear_to_mtr_val': 2.5,
}

ekf = {
    'move_std': 5,
    'measure_std': 1,
}

astolfi = {
    'kp': 4,
    'ka': 30,
    'kb': -4,
}

planning = {
    'radius': 60,
    'spacing': 50,
}