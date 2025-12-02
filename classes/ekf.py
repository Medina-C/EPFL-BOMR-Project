import numpy as np
from numpy import sin, cos

# TODO this uses motor rotational speeds
# should take linear bc thymio docs give us that
class EKF:

    def __init__(self, motion_var, measure_var, state0, robot_wd, dt):
        self.Q = motion_var
        self.R = measure_var
        self.P = self.Q
        self.state = np.array(state0)
        self.T = robot_wd
        self.dt = dt
        self.tag_seen = True

    def generate_next_state(self, inputs, measures):
        # Belief step
        state_bel = self.motion(self.state, inputs)
        
        G = self.motion_jac(state_bel, inputs)
        P_bel = G @ self.P @ G.T + self.Q

        # Update step
        H = self.measure_jac(state_bel)
        K = P_bel @ H.T @ np.linalg.inv(H @ P_bel @ H.T + self.R)
        self.state = state_bel + K @ (measures - self.measure(state_bel))
        self.P = (np.eye(np.size(K, 0)) - K @ H) @ P_bel

        return self.state

    def motion(self, x, u):
        px0, py0, v0, th0, w0 = x[0], x[1], x[2], x[3], x[4]
        vl, vr = u[0], u[1]
        dt, T = self.dt, self.T

        px1 = px0 + v0*cos(th0)*dt
        py1 = py0 + v0*sin(th0)*dt
        v1 = (vl + vr)/2
        th1 = th0 + w0*dt
        w1 = (vr - vl)/T

        return np.array([px1, py1, v1, th1, w1])

    def motion_jac(self, x, u):
        px0, py0, v0, th0, w0 = self.state[0], self.state[1], self.state[2], self.state[3], self.state[4]
        dt, T = self.dt, self.T

        return np.array([
            [1,     0,      cos(th0)*dt,        -v0*sin(th0)*dt,         0],
            [0,     1,      sin(th0)*dt,        v0*cos(th0)*dt,          0],
            [0,     0,      0,                  0,                       0],
            [0,     0,      0,                  1,                       dt],
            [0,     0,      0,                  0,                       0]
        ])
        
    def measure(self, x):
        return np.array([x[0], x[1], x[3]])

    def measure_jac(self, x):
        return np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0],
        ])
    
