from constants import astolfi, robot, general, pid
import numpy as np

class PID:
    def __init__(self):
        self.prev_theta_error = 0
        self.prev_beta_error = 0
        self.sum_theta_error = 0
        self.sum_beta_error = 0
    
    def get_mtr_vels(self, state, goal):
        x0, y0, th0 = state[0], state[1], state[3]
        x1, y1, th1 = goal

        # Manage errors
        theta_error = th0 - th1
        if theta_error > np.pi:
            theta_error = theta_error - 2*np.pi
        self.sum_theta_error += theta_error
        theta_error_diff = (theta_error - self.prev_theta_error)/general['dt']

        angle_to_goal = np.atan2(y1-y0, x1-x0)
        beta_error = th0 - angle_to_goal
        if beta_error > np.pi:
            beta_error = beta_error - 2*np.pi
        self.sum_beta_error += beta_error
        beta_error_diff = (beta_error - self.prev_beta_error)/general['dt']

        # 'Default' settings
        vl = general['drive_vel']
        vr = general['drive_vel']

        # Proportional control
        vl += pid['theta']['kp']*theta_error + pid['beta']['kp']*beta_error
        vr -= pid['theta']['kp']*theta_error + pid['beta']['kp']*beta_error

        # Integral control
        vl += pid['theta']['ki']*self.sum_theta_error + pid['beta']['ki']*self.sum_beta_error
        vr -= pid['theta']['ki']*self.sum_theta_error + pid['beta']['ki']*self.sum_beta_error

        # Derivative control
        vl += pid['theta']['kd']*theta_error_diff + pid['beta']['kd']*beta_error_diff
        vr -= pid['theta']['kd']*theta_error_diff + pid['beta']['kd']*beta_error_diff

        if abs(vl) > general['max_vel']:
            vl = np.sign(vl)*general['max_vel']
        if abs(vr) > general['max_vel']:
            vr = np.sign(vr)*general['max_vel']

        self.prev_theta_error = theta_error
        self.prev_beta_error = beta_error
        # print(vl, vr)
        return vl, vr
    
    def reset_integrals(self):
        self.sum_beta_error = 0
        self.sum_theta_error = 0

def wrap_pi(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi