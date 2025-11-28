import constants as constants
from constants import *  # TODO ugly, change
from classes.ekf import EKF
from classes.motion_control import do_astolfi_cl, PID
from classes.local_avoidance import LocalNav, read_prox
from classes.gp import GlobalPlanner, Plotter
from classes.cv import Vision

import matplotlib.pyplot as plt
import numpy as np
import os
from tdmclient import ClientAsync
from time import time
import cv2 as cv
from enum import Enum


class State(str, Enum):
    GLOBAL_NAV = "global_nav"
    LOCAL_NAV = "local_nav"
    KIDNAPPED = "kidnapped"
    WAIT = "wait"
    STOP = "stop"


def is_lifted(node):
    g = node["prox.ground.delta"]
    return g[0] < 50 and g[1] < 50


def is_grounded(node):
    g = node["prox.ground.delta"]
    return g[0] > 100 and g[1] > 100


state = State.GLOBAL_NAV
a_log = []
theta_cv_log = []
wait_start = None


# Main program
async def main():
    global node, state, wait_start
    await init_tdm()
    
    # TODO CV and path initialization stuff
    vision = Vision(0)
    
    while True:
        vision.update_vision(True, False, False)
        if vision.frame_disp is not None:
            cv.imshow('frame', vision.frame_disp)
        key = cv.waitKey(1) & 0xFF
        if key == 27:  # ESC -> exit
            break
        if key == ord('h'):
            vision.update_vision(True, True, True) 

    vision.get_tag_pose(4)
    vision.get_tag_pose(5)
    cv_data = []
    if vision.obstacles is not None:
        #print(f'obstacles={vision.obstacles}')
        cv_data = vision.obstacles.copy()

    if vision.thymio_pose is not None:
        #print(f'pose={vision.thymio_pose}')
        cv_data.append(vision.thymio_pose['pose_mm'].flatten())

    if vision.target_pose is not None:
        pass
        #print(f'pose={vision.target_pose}')
        # cv_data.append(vision.target_pose['pose_mm'].flatten())
        #print(f'data+thy+tar= {cv_data}')
    cv_data.append(np.array([1100, 800]))
    print(f'final data={cv_data}')

    x = vision.thymio_pose['pose_mm'].flatten()[0]
    y = vision.thymio_pose['pose_mm'].flatten()[1]
    theta = vision.thymio_pose['theta']

    ymax = cv_data[0][1]
    print(f'init before transfo x={x}, y={y}, theta={theta}')
    x, y, theta = cv_coords_to_robot(x, y, ymax, theta)
    print(f'init after transfo x={x}, y={y}, theta={theta}')
    start_pose = (x, y, 0, theta, 0)
    

    goals, plotter = init_global_path(cv_data, ymax)
    g = 0
    goal_pose = goals[g]
    print(f'goal_pose={goal_pose}')

    # class initialization
    ekf = EKF(
        np.diag(constants.ekf['move_vars']), 
        np.diag(constants.ekf['measure_vars']),
        'complex',
        start_pose,
        constants.robot['wheel_base'],
        constants.general['dt']
    )
    pid = PID()
    local_nav = LocalNav()
    state_log = [start_pose]

    nav_ready = True

    # Main loop
    while state != State.STOP:
        #print(state)
        t = time()
        prox = read_prox(node)
        cv.imshow("frame", vision.frame_disp)
        cv.waitKey(1)

        if is_lifted(node):
            state = State.KIDNAPPED
            nav_ready = False
        elif state == State.KIDNAPPED and is_grounded(node):
            state = State.WAIT
            wait_start = time()

        if state == State.KIDNAPPED:
            await set_motors(0, 0)
        elif state == State.WAIT:
            await set_motors(0, 0)
            if wait_start is not None and time() - wait_start > 5.0:
                vision.update_vision(False, True, False)
                vision.get_tag_pose(4)
                x = vision.thymio_pose['pose_mm'].flatten()[0]
                y = vision.thymio_pose['pose_mm'].flatten()[1]
                theta = vision.thymio_pose['theta']
                x, y, theta = cv_coords_to_robot(x, y, ymax, theta)
                start_pose = (x, y, 0, theta, 0)

                cv_data2 = []
                if vision.obstacles is not None:
                    cv_data2 = vision.obstacles.copy()
                cv_data2.append(vision.thymio_pose['pose_mm'].flatten())
                cv_data2.append(cv_data[-1])

                goals, plotter = init_global_path(cv_data2, ymax)
                g = 0
                goal_pose = goals[g]

                ekf = EKF(
                    np.diag(constants.ekf['move_vars']), 
                    np.diag(constants.ekf['measure_vars']),
                    'complex',
                    start_pose,
                    constants.robot['wheel_base'],
                    constants.general['dt']
                )
                pid = PID()
                local_nav = LocalNav()
                state_log.append(start_pose)
                nav_ready = True
                state = State.GLOBAL_NAV
        else:
            if not nav_ready:
                sleep_time = constants.general['dt'] - (time()-t)
                if sleep_time > 0:
                    await client.sleep(sleep_time)
                continue

            current_pose = get_current_pose(ekf, vision, ymax)
            print(current_pose)
            state_log.append(current_pose)

            # Conditions
            if reached_goal(current_pose, goal_pose):
                g += 1
                pid.reset_integrals()
                if g < len(goals):
                    goal_pose = goals[g]
                else:
                    print("Reached goal!")
                    state = State.STOP
                    break
            elif obstacle_detected(prox) and state == State.GLOBAL_NAV:
                state = State.LOCAL_NAV

            # Act based on states
            if state == State.GLOBAL_NAV:
                await global_nav(current_pose, goal_pose, pid)
            elif state == State.LOCAL_NAV:
                cleared = await local_nav.follow_edge(node, prox)
                if cleared:
                    state = State.GLOBAL_NAV
            else:
                print(f"Invalid state {state}!")

        # Processing takes ~100 ms. dt is slightly higher -> wait for the remainder
        sleep_time = constants.general['dt'] - (time()-t)
        #print(sleep_time)
        if sleep_time > 0:
            await client.sleep(sleep_time)

    # Stop
    await set_motors(0, 0)

    vision.stop_video_capture()

    plotter.paths(
        [s[0] for s in state_log],
        [cv_data[0][1] - s[1] for s in state_log]
    )
    plt.plot(theta_cv_log)
    plt.plot([state[3] for state in state_log])
    plt.legend(['theta cv', 'theta state'])
    plt.title("theta cv log")
    # plt.figure()
    # plt.plot(a_log)
    # plt.title("a log")
    plt.show()

# Setup tdm client
async def init_tdm():
    global node, state
    node = await client.wait_for_node()
    await node.lock()
    await node.wait_for_variables()
    await node.watch(variables=True)
    node.add_variables_changed_listener(variables_changed)
    print("Initialized")

def init_global_path(cv_data, ymax):
    planner = GlobalPlanner(cv_data, radius=constants.planning['radius'], spacing=constants.planning['spacing'])
    plotter = Plotter(planner)

    segments_cv = planner.plan()
    plotter.path()
    segments_robot = [
        cv_coords_to_robot(p[0], p[1], ymax, p[2])
        for p in segments_cv
    ]
    return segments_robot, plotter

def cv_coords_to_robot(x, y, ymax, th=None):
    if th is None: 
        return x, ymax-y
    else:
        return x, ymax-y, -th

# Helper function for setting motor values
# takes LINEAR SPEEDS in mm/s as input
async def set_motors(left_vel, right_vel):
    global node

    left_in = int(left_vel * constants.robot['linear_to_mtr_val'])
    right_in = int(right_vel * constants.robot['linear_to_mtr_val'])
    # print(left_in, right_in)

    if left_in > 500:
        left_in = 500
    elif left_in < -500:
        left_in = -500
    if right_in > 500:
        right_in = 500
    elif right_in < -500:
        right_in = -500

    #print(left_in, right_in)

    await node.set_variables({
        "motor.left.target": [left_in],
        "motor.right.target": [right_in],
    })

# Callbacks to run immediately if certain variables are changed
def variables_changed(node, variables):
    global state
    try:
        stop = variables["button.center"]
        if stop:
            state = State.STOP
            print("stop!")
    except KeyError:
        pass
    
theta_cv_log = []

# Use the EKF to update the current estimated pose
# TODO replace measurements with CV measurements
def get_current_pose(ekf: EKF, vision, ymax):
    global node, theta_cv_log
    vl = node['motor.left.speed']
    vr = node['motor.right.speed']
    inputs = np.array([
        vl * constants.robot['mtr_val_to_linear'], 
        vr * constants.robot['mtr_val_to_linear']
    ])
    

    v = (vl+vr)/constants.robot['wheel_base']
    w = (vr-vl)/constants.robot['wheel_base']
    tt = time()
    vision.update_vision(False, False, False)
    #print(f'time for vision upd={time() - tt}')
    tt = time()
    vision.get_tag_pose(4)
    
    #print(f'time for tag pose={time() - tt}')
    x = vision.thymio_pose['pose_mm'].flatten()[0]
    y = vision.thymio_pose['pose_mm'].flatten()[1]
    theta = vision.thymio_pose['theta']
    x, y, theta = cv_coords_to_robot(x, y, ymax, theta)
    #print(f'[get_cur_pose] x={x} y={y} theta={theta}')

    theta_cv_log.append(theta)

    velocity_data = vision.get_tag_velocity(4)

    measurements = [
        x,
        y,
        velocity_data['speed_mm_s'],
        theta,
        velocity_data['omega_rad_s']
    ]

    return ekf.generate_next_state(inputs, measurements)

def reached_goal(current, goal):
    px, py = current[0], current[1]
    gx, gy = goal[0], goal[1]

    # If we're at both coordinates (within tolerance)
    # then we've reached the goal (regardless of angle)
    return (abs(px - gx) < constants.general['goal_buffer'] and
        abs(py - gy) < constants.general['goal_buffer'])

def obstacle_detected(prox):
    front = prox[IDX_FRONT]
    left = max(prox[IDX_NO], prox[IDX_NNO])
    right = max(prox[IDX_NNE], prox[IDX_NE])
    front_left = prox[IDX_NNO]
    front_right = prox[IDX_NNE]
        
    return front >= FRONT_THRESHOLD or front_left >= FRONT_THRESHOLD or front_right >= FRONT_THRESHOLD

async def global_nav(start_pose, goal_pose, pid: PID):
    global a_log
    #vl, vr, a = do_astolfi_cl(start_pose, goal_pose)
    #a_log.append(a)
    vl, vr = pid.get_mtr_vels(start_pose, goal_pose)
    tt = time()
    await set_motors(vl, vr)
    #print(f'time for set mot={time()-tt}')

if __name__ == "__main__":
    print("Hello, World!")
    client = ClientAsync()
    client.run_async_program(main)
    client.close()
    print("stopped")
    print(state)

    os._exit(0)
