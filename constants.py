'''Stores all the constants we need,
   so main.py isn't as cluttered'''

general = {
    'dt': 0.18,
    'goal_buffer': 40,
    'drive_vel': 130*2/3,
    'max_vel': 130*2/3,
    'max_accel': 20000,
    'max_w': 0.7,
}

robot = {
    'wheel_rad': 44/2,
    'wheel_base': 95,
    'mtr_val_to_linear': 0.3664,
    'linear_to_mtr_val': 1/0.3664,
}

ekf = {
    'move_vars': [4.48+0.364, 4.48+0.213, 10.78, 0.0156+0.00051, 0.0088],
    'measure_vars': [0.364, 0.213, 0.00051],
}

pid = {
    'theta': {  
        'kp': 2*2/3,
        'ki': 0,
        'kd': 0, 
    },
    'beta': {
        'kp': 80*2/3,
        'ki': 2/3,
        'kd': 2/3,
    }
}

planning = {
    'radius': 100,
    'spacing': 10000,
}

tag_ids = {
    'thymio': 4,
    'target': 5
}

kidnapping = {
    "lifted_threshold": 50,
    "grounded_threshold": 100,
    "wait_time": 4
}

# Sensor indices for prox.horizontal (front arc only)
IDX_NO = 0
IDX_NNO = 1
IDX_FRONT = 2
IDX_NNE = 3
IDX_NE = 4

local_avoidance = {
    'front_threshold': 1600,
    'clear_threshold': 800,
    'check_interval': 0.05,
    'forward_speed': 160,
    'turn_speed': 130,
    'proportional_gain': 0.01,
    'target_side_value': 1200,
    'exit_consistent_count': 5,
    'detection_threshold': 1800,
    'side_threshold': 1000,
}

# Adrien CV
MAP_W = 1260    #in mm
MAP_H = 890     #in mm
THYMIO_ID = 4
TARGET_ID = 5


A4_W = 210  #in mm
A4_H = 297  #in mm
IRL_MAP_DIST_W = 4*A4_W - 28    #in mm

DISP_FRAME_W = 960
DISP_FRAME_H = 540