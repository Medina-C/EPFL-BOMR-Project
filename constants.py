'''Stores all the constants we need,
   so main.py isn't as cluttered'''

general = {
    'dt': 0.22,
    'goal_buffer': 50,
    'drive_vel': 130,
    'max_vel': 130,
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
    'move_vars': [250, 250, 125, 500, 500],
    'measure_vars': [1, 1, 2, 1, 2],
}

astolfi = {
    'kp': 4,
    'ka': 80,
    'kb': -2,
}

pid = {
    'theta': {  
        'kp': 2,
        'ki': 0,
        'kd': 0, 
    },
    'beta': {
        'kp': 80,
        'ki': 0,
        'kd': 0,
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
    "wait_time": 5
}

IDX_NO = 0
IDX_NNO = 1
IDX_FRONT = 2
IDX_NNE = 3
IDX_NE = 4
DETECTION_THRESHOLD = 1800   # obstacle detection threshold for all sensors (lower = earlier detection when farther away)
SIDE_THRESHOLD = 1000
CLEAR_THRESHOLD = 800
CHECK_INTERVAL = 0.05    # seconds between loop iterations
FORWARD_SPEED = 160      # forward speed
TURN_SPEED = 160         # turning base speed used in corrections
PROPORTIONAL_GAIN = 0.0005  # proportional gain for side error -> motor differential (reduced to minimize turning)
TARGET_SIDE_VALUE = 1000  # desired side prox reading while following (tunable)
EXIT_CONSISTENT_COUNT = 6  # number of consecutive "clear" readings to exit
