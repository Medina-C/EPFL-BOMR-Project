import constants as constants
from classes.ekf import EKF
from classes.motion_control import *

import numpy as np
import matplotlib.pyplot as plt
import random

'''
Runs a simple open-loop simulation with an EKF, noisy measurements, 
and noisy motion. 
'''

start_pose = (0, 0, 0, np.pi/4, 0)
goals = [
    (0, 200, np.pi/2),
    (0, 400, np.pi/2),
    (150, 550, np.pi/4),
    (300, 700, np.pi/4),
    (500, 700, 0)
]

# EKF initialization
ekf = EKF(
    np.diag(constants.ekf['move_vars']), 
    np.diag(constants.ekf['measure_vars']),
    'complex',
    start_pose,
    constants.robot['wheel_base'],
    constants.general['dt']
)

# Logging EKF estimation, target state, and real state (with noise)
log_est = [np.array(start_pose)]
log_est_waypts = [np.array(start_pose)]
log_real = [np.array(start_pose)]
log_ideal = [np.array(start_pose)]
log_u = []

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

error_log = []

pid = PID()

for goal in goals:
    #u_set, waypts = do_astolfi_ol(est_state, goal)
    n = 0
    while not reached_goal(est_state, goal) and n < 100:
        n += 1
        u = pid.get_mtr_vels(est_state, goal)
        log_u.append(u)
        print(u)

        z_nonoise = [log_real[-1][0], log_real[-1][1], log_real[-1][3]]
        z = [var + random.gauss(0, 0.1) for var in z_nonoise] # Generate measurement with noise
        est_state = ekf.generate_next_state(u, z)   # Use EKF to estimate next state
        log_est.append(est_state)

        # Update real position (with noise - motion model is imperfect in real life)
        no_noise_state = ekf.motion(log_real[-1], u)
        real_state = np.array([var + random.gauss(0, 0.0001) for var in no_noise_state])
        log_real.append(real_state) 

        ideal_state = ekf.motion(log_ideal[-1], u)
        log_ideal.append(ideal_state)

    log_ideal.append(est_state)
    log_est_waypts.append(est_state)

#print(np.mean(error_log))

plt.plot([x[0] for x in log_est], [x[1] for x in log_est], c='g')
plt.plot([x[0] for x in log_real], [x[1] for x in log_real], c='r')
#plt.plot([x[0] for x in log_ideal], [x[1] for x in log_ideal], c='c')
plt.scatter([x[0] for x in goals], [x[1] for x in goals], c='c')
plt.scatter([x[0] for x in log_est_waypts], [x[1] for x in log_est_waypts], c='r')
plt.axis('equal')
#plt.legend(["Estimated state", "Real State", "Target Waypoints", "Real Waypoints"])


plt.figure()
# plt.plot([u[0] for u in log_u])
# plt.plot([u[1] for u in log_u])
# plt.legend(['Left wheel velocity', 'Right wheel velocity'])
plt.plot([x[3] for x in log_est])

plt.show()