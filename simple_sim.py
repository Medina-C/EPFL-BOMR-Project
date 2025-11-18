import constants
from ekf import EKF
from motion_control import do_astolfi

import numpy as np
import matplotlib.pyplot as plt
import random

'''
Runs a simple open-loop simulation with an EKF, noisy measurements, 
and noisy motion. 
'''

pts = 30
version = 'basic'

start_pose = (0, 0, np.pi/2)
goals = []
for i in range(pts):
    goals.append((200/pts*(i+1), 0, 0))

# EKF initialization
ekf = EKF(
    (constants.ekf['move_std']**2)*np.eye(len(start_pose)), 
    (constants.ekf['measure_std']**2)*np.eye(len(start_pose)),
    'basic',
    start_pose,
    constants.robot['wheel_base'],
    constants.general['dt']
)

# Logging EKF estimation, target state, and real state (with noise)
log_est = [np.array(start_pose)]
log_est_waypts = [np.array(start_pose)]
log_real = [np.array(start_pose)]

def reached_goal(current, goal):
    px, py = current[0], current[1]
    gx, gy = goal[0], goal[1]

    # If we're at both coordinates (within tolerance)
    # then we've reached the goal (regardless of angle)
    return (abs(px - gx) < constants.general['goal_buffer'] and
        abs(py - gy) < constants.general['goal_buffer'])

# Run simulation
cur_goal = 0
est_state = start_pose

for goal in goals:
    u_set = do_astolfi(est_state, goal)
    for u in u_set:

        z = [var + random.gauss(0, constants.ekf['measure_std']) for var in log_real[-1]] # Generate measurement with noise
        est_state = ekf.generate_next_state(u, z)   # Use EKF to estimate next state
        log_est.append(est_state)

        # Update real position (with noise - motion model is imperfect in real life)
        no_noise_state = ekf.motion(log_real[-1], u)
        real_state = np.array([var + random.gauss(0, constants.ekf['move_std']) for var in no_noise_state])
        log_real.append(real_state) 
    log_est_waypts.append(est_state)


plt.plot([x[0] for x in log_est], [x[1] for x in log_est], c='g')
plt.plot([x[0] for x in log_real], [x[1] for x in log_real], c='r')
plt.scatter([x[0] for x in goals], [x[1] for x in goals])
plt.scatter([x[0] for x in log_est_waypts], [x[1] for x in log_est_waypts], c='r')
plt.axis('equal')
plt.legend(["Estimated state", "Real State", "Target Waypoints", "Real Waypoints"])
plt.show()