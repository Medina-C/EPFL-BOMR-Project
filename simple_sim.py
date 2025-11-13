from ekf import EKF
import numpy as np
import matplotlib.pyplot as plt
import random

N = 1000
version = 'basic'

# Initial state
if version == 'basic':
    state = np.array([0, 0, 0])
else:
    state = np.array([0, 0, 0, 0, 0])

# Constants
move_std = 0.01
measure_std = 0.05

wheel_rad = 0.01
robot_wd = 0.1
dt = 0.01

# Set up EKF
ekf = EKF(
    (move_std**2)*np.eye(len(state)), 
    (measure_std**2)*np.eye(len(state)),
    version,
    state,
    wheel_rad,
    robot_wd,
    dt
)

# Assume constant input for now
u = np.array([40, 45])

# Logging EKF estimation, target state, and real state (with noise)
log_est = [state]
log_ideal = [state]
log_real = [state]

# Run simulation
for i in range(N):
    z = [var + random.gauss(0, measure_std) for var in log_real[-1]] # Generate measurement with noise
    est_state = ekf.generate_next_state(u, z)   # Use EKF to estimate next state
    log_est.append(est_state)

    # Update real position (with noise - motion model is imperfect in real life)
    no_noise_state = ekf.motion(log_real[-1], u)
    real_state = np.array([var + random.gauss(0, move_std) for var in no_noise_state])
    log_real.append(real_state)    
    # Update target state
    log_ideal.append(ekf.motion(log_ideal[-1], u))    

plt.plot([x[0] for x in log_est], [x[1] for x in log_est])
plt.plot([x[0] for x in log_real], [x[1] for x in log_real])
plt.plot([x[0] for x in log_ideal], [x[1] for x in log_ideal])
plt.axis('equal')
plt.legend(["Estimated State", "Real State", "Ideal State"])
plt.show()