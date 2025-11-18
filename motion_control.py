from constants import astolfi, robot, general
import numpy as np

def do_astolfi(start, goal):
    x0, y0, th0 = start
    x1, y1, th1 = goal

    dx = x1 - x0
    dy = y1 - y0

    inputs = []

    # Initial p, a, b
    p = np.sqrt(dx**2 + dy**2)
    a = -th0 + np.arctan2(dy, dx)
    b = (th1 - th0) - a

    while p > general['goal_buffer']:
        # Convert from p, a, b to velocities
        v = astolfi['kp']*p
        w = astolfi['ka']*a + astolfi['kb']*b

        vl = v + w*robot['wheel_base']/2
        vr = v - w*robot['wheel_base']/2

        inputs.append((vl, vr))

        # Update p, a, b
        dp = -astolfi['kp']*p*np.cos(a)
        da = astolfi['kp']*np.sin(a) - astolfi['ka']*a - astolfi['kb']*b
        db = -astolfi['kp']*np.sin(a)

        p = p + dp*general['dt']
        a = a + da*general['dt']
        b = b + db*general['dt']

    return inputs
