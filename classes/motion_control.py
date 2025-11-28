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

def do_astolfi_cl(state, goal):
    # Extract parameters from state
    p, a, b = calc_params([state[0], state[1], state[3]], goal)

    prev_v, prev_w = state[2], state[4]
    prev_vl = prev_v - robot['wheel_base']/2*prev_w
    prev_vr = prev_v + robot['wheel_base']/2*prev_w

    # Convert from p, a, b to next velocities
    v = general['drive_vel']
    w_prelim = astolfi['ka']*a + astolfi['kb']*b

    # Maximum angular velocity based on max wheel speeds
    w = np.sign(w_prelim) * min(
        abs(w_prelim), 
        general['max_w']#(general['max_vel'] - general['drive_vel'])/robot['wheel_base']*2
    )

    vl = v - w*robot['wheel_base']/2
    vr = v + w*robot['wheel_base']/2

    # Check acceleration, update velocities if necessary
    dvl = abs(prev_vl - vl)
    dvr = abs(prev_vr - vr)
    max_dv = general['max_accel']*general['dt']

    updated = False
    if dvl > max_dv:
        vl = prev_vl + np.sign(vl-prev_vl)*max_dv
        updated = True
    if dvr > max_dv:
        vr = prev_vr + np.sign(vr-prev_vr)*max_dv
        updated = True
    if updated:
        v = (vl+vr)/robot['wheel_base']
        w = (vr-vl)/robot['wheel_base']

    return vl, vr, a

def wrap_pi(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi

def calc_params(start, goal):
    x0, y0, th0 = start
    x1, y1, th1 = goal

    dx = x1 - x0
    dy = y1 - y0

    # Initial p, a, b
    p = np.sqrt(dx**2 + dy**2)
    a = wrap_pi(-th0 + np.arctan2(dy, dx))
    
    # if a >= np.pi:
    #     a -= 2*np.pi
    # elif a <= -np.pi:
    #     a += 2*np.pi
    
    b = (th1 - th0) - a

    return p, a, b

def do_astolfi_ol(start, goal):
    p, a, b = calc_params([start[0], start[1], start[3]], goal)

    inputs = []
    waypts = [(p, a, b)]

    prev_vl = 0
    prev_vr = 0

    while p > general['goal_buffer']:
        # Convert from p, a, b to velocities
        v = general['drive_vel']
        w_prelim = astolfi['theta']['ka']*a + astolfi['theta']['kb']*b

        w = np.sign(w_prelim) * min(
            abs(w_prelim), 
            general['max_w']#(general['max_vel'] - general['drive_vel'])/robot['wheel_base']*2
        )
    
        vl = v - w*robot['wheel_base']/2
        vr = v + w*robot['wheel_base']/2

        # Check acceleration, update velocities if necessary
        dvl = abs(prev_vl - vl)
        dvr = abs(prev_vr - vr)
        max_dv = general['max_accel']*general['dt']

        updated = False
        if dvl > max_dv:
            vl = prev_vl + np.sign(vl-prev_vl)*max_dv
            updated = True
        if dvr > max_dv:
            vr = prev_vr + np.sign(vr-prev_vr)*max_dv
            updated = True
        if updated:
            v = (vl+vr)/robot['wheel_base']
            w = (vr-vl)/robot['wheel_base']
        #print(vl, vr, updated)

        inputs.append((vl, vr))
        prev_vl = vl
        prev_vr = vr

        # Update p, a, b
        dp = -v*np.cos(a)
        da = v*np.sin(a)/p - w
        db = -v*np.sin(a)/p

        p = p + dp*general['dt']
        a = a + da*general['dt']
        b = b + db*general['dt']
        waypts.append((p, a, b))

    return inputs

# def calc_error(cur_pos, ideal_wp, goal):
#     dx = goal[0] - cur_pos[0]
#     dy = goal[1] - cur_pos[1]
#     th0 = cur_pos[2]
#     th1 = goal[2]
    
#     p_ideal, a_ideal, b_ideal = ideal_wp

#     p_real = np.sqrt(dx**2 + dy**2)
#     a_real = -th0 + np.arctan2(dy, dx)
#     b_real = (th1 - th0) - a_real

#     p_error = p_real - p_ideal
#     a_error = a_real - a_ideal
#     b_error = b_real - b_ideal

#     return (p_error, a_error, b_error)

# def do_pid(errors, inputs):
#     vl, vr = inputs

#     vl += errors[0]*pid['p'][0]
#     vl -= errors[1]*pid['p'][1]
#     vl -= errors[1]*pid['p'][2]

#     vr += errors[0]*pid['p'][0]
#     vr += errors[1]*pid['p'][1]
#     vr += errors[1]*pid['p'][2]

#     return vl, vr

