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

tag_ids = {
    'thymio': 4,
    'target': 5
}

IDX_NO = 0
IDX_NNO = 1
IDX_FRONT = 2
IDX_NNE = 3
IDX_NE = 4
FRONT_THRESHOLD = 1300   # detect obstacle early (increased for earlier detection of spontaneous obstacles)
SIDE_THRESHOLD = 1000
CLEAR_THRESHOLD = 800
CHECK_INTERVAL = 0.05    # seconds between loop iterations
FORWARD_SPEED = 160      # forward speed
TURN_SPEED = 160         # turning base speed used in corrections
PROPORTIONAL_GAIN = 0.0005  # proportional gain for side error -> motor differential (reduced to minimize turning)
TARGET_SIDE_VALUE = 1000  # desired side prox reading while following (tunable)
EXIT_CONSISTENT_COUNT = 6  # number of consecutive "clear" readings to exit
