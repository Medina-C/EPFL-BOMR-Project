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

pts = 10
version = 'basic'

start_pose = (0, 0, -0.01)
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
n = 0
est_state = start_pose
while(n < 100):
    if n == 0 or reached_goal(est_state, goals[cur_goal]):
        cur_goal += 1
        if cur_goal == len(goals):
            print("mission complete!")
            break

        u = do_astolfi(log_est[-1], goals[cur_goal])
        print(u)

    z = [var + random.gauss(0, constants.ekf['measure_std']) for var in log_real[-1]] # Generate measurement with noise
    est_state = ekf.generate_next_state(u, z)   # Use EKF to estimate next state
    log_est.append(est_state)

    # Update real position (with noise - motion model is imperfect in real life)
    no_noise_state = ekf.motion(log_real[-1], u)
    real_state = np.array([var + random.gauss(0, constants.ekf['move_std']) for var in no_noise_state])
    log_real.append(real_state) 

    n += 1     

if n == 1000:
    print("mission failed")  

plt.plot([x[0] for x in log_est], [x[1] for x in log_est])
plt.scatter([x[0] for x in log_real], [x[1] for x in log_real])
plt.scatter([x[0] for x in goals], [x[1] for x in goals])
plt.axis('equal')
plt.legend(["Estimated State", "Real State", "Ideal State"])
plt.show()