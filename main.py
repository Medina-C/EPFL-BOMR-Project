import constants
from constants import * 
from classes.ekf import EKF
from classes.motion_control import PID
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

# Enum for storing robot states
class State(str, Enum):
    GLOBAL_NAV = "global_nav" # Follow global path
    LOCAL_NAV = "local_nav" # Avoid local obsacle
    KIDNAPPED = "kidnapped" # Kidnapped (currently in the air)
    WAIT = "wait" # Wait time after kidnapping
    STOP = "stop" # Program over

# Initialize global variables
state = State.GLOBAL_NAV
wait_start = None

# Main program
async def main():
    global node, state, wait_start
    await init_tdm() # Initialize Thymio
    
    vision = Vision(1)
    cv.namedWindow('frame', cv.WINDOW_NORMAL)
    cv.resizeWindow('frame', DISP_FRAME_W, DISP_FRAME_H)

    start_pose, goals, goals_cv, plotter, cv_data, ymax = init_vision(vision)

    g = 0
    goal_pose = goals[g]

    # class initialization
    ekf = EKF(
        np.diag(constants.ekf['move_vars']), 
        np.diag(constants.ekf['measure_vars']),
        start_pose,
        constants.robot['wheel_base'],
        constants.general['dt']
    )
    pid = PID()
    local_nav = LocalNav()
    state_log = [start_pose]
    cov_matrix = ekf.P[0:2, 0:2]

    nav_ready = True

    # Main loop
    while state != State.STOP:
        t = time()
        prox = read_prox(node)
        
        vision.update_disp_frame(state_log, goals_cv, cv_data[0][1], cov_matrix)        
        cv.imshow('frame', vision.frame_disp)
        cv.resizeWindow('frame', (1920, 1080))
        cv.waitKey(1)

        # Check for kidnapping conditions, that overrides path following
        if is_lifted():
            state = State.KIDNAPPED
            nav_ready = False
        # Check if we've been un-kidnapped
        elif state == State.KIDNAPPED and is_grounded():
            state = State.WAIT
            await client.sleep(1)
            vision.update_vision(True, True, True)
            wait_start = time()

        # If kidnapped, stop
        if state == State.KIDNAPPED:
            await set_motors(0, 0)
        # If we're set back down, delay for a small time and then continue
        elif state == State.WAIT:
            await set_motors(0, 0)
            if wait_start is not None and time() - wait_start > constants.kidnapping["wait_time"]:
                goals_cv, goal_pose, ekf, pid, local_nav, nav_ready, plotter = await wait_after_kidnapping(vision, ymax, cv_data)
        # If not kidnapped, run normal behaviours
        else:
            # Make sure everything is reset after kidnapping
            if not nav_ready:
                sleep_time = constants.general['dt'] - (time()-t)
                if sleep_time > 0:
                    await client.sleep(sleep_time)
                continue
            
            # Get and log current pose
            current_pose, cov_matrix = get_current_pose(ekf, vision, ymax)
            state_log.append(current_pose)

            # If we've reached the next waypoint, update to the next
            # If this was the last, we're done!
            if reached_goal(current_pose, goal_pose):
                g += 1
                pid.reset_integrals()
                if g < len(goals):
                    goal_pose = goals[g]
                else:
                    print("Reached goal!")
                    state = State.STOP
                    break
            # If there's an obstacle, switch to local avoidance mode
            elif obstacle_detected(prox) and state == State.GLOBAL_NAV:
                state = State.LOCAL_NAV

            # Act based on states
            if state == State.GLOBAL_NAV:
                await global_nav(current_pose, goal_pose, pid)
            elif state == State.LOCAL_NAV:
                cleared = await local_nav.follow_edge(node, prox)
                if cleared:
                    state = State.GLOBAL_NAV
                    print("cleared local nav")
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

    pid.plot()

# Setup tdm client
async def init_tdm():
    global node, state
    node = await client.wait_for_node()
    await node.lock()
    await node.wait_for_variables()
    await node.watch(variables=True)
    node.add_variables_changed_listener(variables_changed)
    print("Initialized")

# Initialize computer vision system
def init_vision(vision):
    global node, state
    goals = None
    
    # Repeat initialization until we've successfully found a path
    while goals is None:
        # Show image until we tell it to exit
        while True:
            # Displau frame
            vision.update_vision(False, False, False)
            if vision.frame_disp is not None:
                cv.imshow('frame', vision.frame_disp)
                cv.resizeWindow('frame', (DISP_FRAME_W, DISP_FRAME_H))

            # Exit and start program
            key = cv.waitKey(1) & 0xFF
            if key == 27:  # ESC -> exit
                break
            # Get obstacles
            if key == ord('h'):
                vision.update_vision(True, True, True) 

        vision.get_tag_pose(constants.tag_ids['thymio'])
        vision.get_tag_pose(constants.tag_ids['goal'])
        cv_data = []

        # Get obstacles and start/end poses from CV
        if vision.obstacles is None or vision.thymio_pose is None or vision.target_pose is None:
            print('[VISION] error getting cv_data')
        else:
            cv_data = vision.obstacles.copy()
            cv_data.append(vision.thymio_pose['pose_mm'].flatten())
            cv_data.append(vision.target_pose['pose_mm'].flatten())

        x = vision.thymio_pose['pose_mm'].flatten()[0]
        y = vision.thymio_pose['pose_mm'].flatten()[1]
        theta = vision.thymio_pose['theta']

        # Convert coordinates to robot frame
        ymax = cv_data[0][1]
        x, y, theta = cv_coords_to_robot(x, y, ymax, theta)
        start_pose = (x, y, 0, theta, 0)
        
        # Get global path
        goals_cv, goals, plotter = init_global_path(cv_data, ymax)

    return start_pose, goals_cv, goals, plotter, cv_data, ymax

def init_global_path(cv_data, ymax):
    planner = GlobalPlanner(cv_data, radius=constants.planning['radius'], spacing=constants.planning['spacing'])
    plotter = Plotter(planner)

    segments_cv = planner.plan()
    if segments_cv is None:
        return None, None, None
    plotter.path()
    segments_robot = [
        cv_coords_to_robot(p[0], p[1], ymax, p[2])
        for p in segments_cv
    ]
    return segments_cv, segments_robot, plotter

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
    vl = node['motor.left.speed'] * constants.robot['mtr_val_to_linear']
    vr = node['motor.right.speed'] * constants.robot['mtr_val_to_linear']
    inputs = np.array([
        vl , 
        vr
    ])

    v = (vl + vr) / 2
    w = (vr - vl) / constants.robot['wheel_base']

    vision.update_vision(False, False, False)
    tag_seen = vision.get_tag_pose(4)

    if (not tag_seen) and (ekf.tag_seen):
        ekf.tag_seen = False
        ekf.R = np.eye(3)*999999
        #print(f'Tag hidden, setting R={ekf.R}')
    elif (tag_seen) and (not ekf.tag_seen):
        ekf.tag_seen = True
        ekf.R = np.diag(constants.ekf['measure_vars'])
        #print(f"Tag visible, setting R={ekf.R}")

    x = vision.thymio_pose['pose_mm'].flatten()[0]
    y = vision.thymio_pose['pose_mm'].flatten()[1]
    theta = vision.thymio_pose['theta']
    x, y, theta = cv_coords_to_robot(x, y, ymax, theta)

    theta_cv_log.append(theta)

    measurements = [
        x,
        y,
        theta,
    ]

    state_estimate, cov_matrix = ekf.generate_next_state(inputs, measurements)

    return state_estimate, cov_matrix

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
    far_left = prox[IDX_NO]  # Far left sensor
    far_right = prox[IDX_NE]  # Far right sensor
        
    # Check all front-facing sensors including far left and far right
    detection_threshold = constants.local_avoidance['detection_threshold']
    return (front >= detection_threshold or 
            front_left >= detection_threshold or 
            front_right >= detection_threshold or
            far_left >= detection_threshold or
            far_right >= detection_threshold)

async def global_nav(start_pose, goal_pose, pid: PID):
    global a_log
    #vl, vr, a = do_astolfi_cl(start_pose, goal_pose)
    #a_log.append(a)
    vl, vr = pid.get_mtr_vels(start_pose, goal_pose)
    tt = time()
    await set_motors(vl, vr)
    #print(f'time for set mot={time()-tt}')

def is_lifted():
    global node
    g = node["prox.ground.delta"]
    return g[0] < constants.kidnapping["lifted_threshold"] and g[1] < constants.kidnapping["lifted_threshold"]

def is_grounded():
    global node
    g = node["prox.ground.delta"]
    return g[0] > constants.kidnapping["grounded_threshold"] and g[1] > constants.kidnapping["grounded_threshold"]

async def wait_after_kidnapping(vision, ymax, cv_data):
    global node, state
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

    goals_cv, goals, plotter = init_global_path(cv_data2, ymax)
    g = 0
    goal_pose = goals[g]

    ekf = EKF(
        np.diag(constants.ekf['move_vars']), 
        np.diag(constants.ekf['measure_vars']),
        start_pose,
        constants.robot['wheel_base'],
        constants.general['dt']
    )
    pid = PID()
    local_nav = LocalNav()
    nav_ready = True
    state = State.GLOBAL_NAV

    return goals_cv, goal_pose, ekf, pid, local_nav, nav_ready, plotter

if __name__ == "__main__":
    print("Hello, World!")
    client = ClientAsync()
    client.run_async_program(main)
    client.close()
    print("Stopped")
    print(state)

    os._exit(0)
