from constants import astolfi, robot, general
import numpy as np

def do_astolfi(start, goal):
    x0, y0, th0 = start
    x1, y1, th1 = goal

    dx = x1 - x0
    dy = y1 - y0

    p = np.sqrt(dx**2 + dy**2)
    a = -th0 + np.arctan2(dy, dx)
    b = (th1 - th0) - a
    print(p, a, b)

    v = astolfi['kp']*p
    w = astolfi['ka']*a + astolfi['kb']*b

    vl = v - w*robot['wheel_base']/2
    vr = v + w*robot['wheel_base']/2

    # Normalize
    avg_vel = (abs(vl)+abs(vr))/2
    vl = vl / avg_vel * general['drive_vel']
    vr = vr / avg_vel * general['drive_vel']

    return vl, vr
