import numpy as np
from numpy import sin, cos

# TODO this uses motor rotational speeds
# should take linear bc thymio docs give us that
class EKF:

    def __init__(self, motion_var, measure_var, model, state0, robot_wd, dt):
        self.Q = motion_var
        self.R = measure_var
        self.P = self.R
        self.state = np.array(state0)
        self.T = robot_wd
        self.dt = dt
        self.model = model

    def generate_next_state(self, inputs, measures):
        # Belief step
        state_bel = self.motion(self.state, inputs)
        
        G = self.motion_jac(self.state, inputs)
        P_bel = G @ self.P @ G.T + self.Q

        # Update step
        H = self.measure_jac(self.state)
        K = P_bel @ H.T @ np.linalg.inv(H @ P_bel @ H.T + self.R)
        self.state = state_bel + K @ (measures - self.measure(self.state))
        self.P = (np.eye(np.size(K, 0)) - K @ H) @ P_bel

        return self.state


    def motion_basic(self, x, u):
        px0, py0, th0 = x[0], x[1], x[2]
        vl, vr = u[0], u[1]
        dt, T = self.dt, self.T

        v = (vl + vr) / 2
        w = (vl - vr) / T

        px1 = px0 + v*cos(th0)*dt
        py1 = py0 + v*sin(th0)*dt
        th1 = th0 + w*dt

        return np.array([px1, py1, th1])

    def motion_jac_basic(self, x, u):
        px0, py0, th0 = x[0], x[1], x[2]
        vl, vr = u[0], u[1]
        dt, T = self.dt, self.T

        v = (vl + vr) / 2
        w = (vl - vr) / T

        return np.array([
            [1,     0,      -v*sin(th0)*dt],
            [0,     1,      v*cos(th0)*dt],
            [0,     0,      1]
        ])

    def measure_basic(self, x):
        return x

    def measure_jac_basic(self, x):
        return np.array([
            [1, 0, 0], 
            [0, 1, 0],
            [0, 0, 1]
        ])

    def motion_complex(self, x, u):
        px0, py0, v0, th0, w0 = x[0], x[1], x[2], x[3], x[4]
        vl, vr = u[0], u[1]
        dt, T = self.dt, self.T

        px1 = px0 + v0*cos(th0)*dt
        py1 = py0 + v0*sin(th0)*dt
        v1 = (vl + vr)/2
        th1 = th0 + w0*dt
        w1 = (vr - vl)/T

        return np.array([px1, py1, v1, th1, w1])

    def motion_jac_complex(self, x, u):
        px0, py0, v0, th0, w0 = self.state[0], self.state[1], self.state[2], self.state[3], self.state[4]
        vl, vr = u[0], u[1]
        dt, T = self.dt, self.T

        return np.array([
            [1,     0,      -v0*sin(th0)*dt,    cos(th0)*dt,    0],
            [0,     1,      v0*cos(th0)*dt,     sin(th0)*dt,    0],
            [0,     0,      1,                  0,              dt],
            [0,     0,      0,                  0,              0],
            [0,     0,      0,                  0,              0]
        ])
        
    def measure_complex(self, x):
        return x

    def measure_jac_complex(self, x):
        return np.eye(len(x))

    def motion(self, x, u):
        if self.model=='basic':
            return self.motion_basic(x, u)
        else:
            return self.motion_complex(x, u)
        
    def motion_jac(self, x, u) -> np.ndarray:
        if self.model=='basic':
            return self.motion_jac_basic(x, u)
        else:
            return self.motion_jac_complex(x, u)
        
    def measure(self, x):
        if self.model=='basic':
            return self.measure_basic(x)
        else:
            return self.measure_complex(x)
        
    def measure_jac(self, x) -> np.ndarray:
        if self.model=='basic':
            return self.measure_jac_basic(x)
        else:
            return self.measure_jac_complex(x)
    
