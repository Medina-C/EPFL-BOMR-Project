import constants as constants
from constants import * #TODO ugly, change
from classes.ekf import EKF
from classes.motion_control import do_astolfi_cl, PID
from classes.local_avoidance import LocalNav, read_prox

from classes.gp import GlobalPlanner, Plotter
from classes.cv import Vision

import matplotlib.pyplot as plt
import numpy as np
import os
from tdmclient import ClientAsync
from time import time, sleep
import cv2 as cv

# Use CV to get the real velocity <-> motor input conversion
# More accurate than using a tape measure, and lets us skip the acceleration
async def test_velocity_to_motor_input():
    global node
    await init_tdm()

    vision = Vision(1)
     
    while True:
        vision.update_vision(True, False, False)
        if vision.frame_disp is not None:
            cv.imshow('frame', vision.frame_disp)
        key = cv.waitKey(1) & 0xFF
        if key == 27:  # ESC -> exit
            break
        if key == ord('h'):
            vision.update_vision(True, True, True) 

    output = open('outputs/speed_to_mtr_input.csv', 'a')

    delay = 10

    for speed in [100]:

        v = {
            "motor.left.target": [speed],
            "motor.right.target": [speed],
        }
        await node.set_variables(v)

        # Wait for acceleration
        await client.sleep(1)
        vision.update_vision(False, False, False)
        sleep(0.01)
        vision.get_tag_pose(4)
        sleep(0.01)
        start_pose = vision.thymio_pose['pose_mm'].flatten()
        print(start_pose)

        sleep(1000)

        print('here')
        vision.update_vision(False, False, False)
        sleep(0.01)
        vision.get_tag_pose(4)
        sleep(0.01)
        cv.imshow('framee', vision.frame_disp)
        cv.waitKey(1)
        end_pose = vision.thymio_pose['pose_mm'].flatten()
        print(end_pose)

        v = {
            "motor.left.target": [0],
            "motor.right.target": [0],
        }
        await node.set_variables(v)
        await client.sleep(1)

        distance = np.sqrt((end_pose[0]-start_pose[0])**2 + (end_pose[1]-start_pose[1])**2)
        velocity = distance/delay

        output.write(f"{speed}, {velocity}\n")

    output.close()

# Plot velocity profile to see how long acceleration takes
async def observe_accel():
    global node
    await init_tdm()

    speed = 400

    v = {
        "motor.left.target": [speed],
        "motor.right.target": [speed],
    }
    await node.set_variables(v)
    print(node['motor.left.target'])
    print(node["motor.right.target"])

    start = time()
    logs = [[], []]
    x = []

    while time()-start < 5: 
        x.append(time()-start)
        logs[0].append(node["motor.left.speed"])
        logs[1].append(node["motor.right.speed"])
        await client.sleep(0.05)

    v = {
        "motor.left.target": [0],
        "motor.right.target": [0],
    }
    await node.set_variables(v)

    plt.plot(x, logs[0])
    plt.plot(x, logs[1])
    plt.legend(["left", "right"])
    plt.show()
    
async def test_linear_variances():
    global node
    await init_tdm()

    N = 500
    target = 130*constants.robot['linear_to_mtr_val']
    T = 0.05

    vel_error = []
    rdata = []
    ldata = []

    v = {
        "motor.left.target": [target],
        "motor.right.target": [target],
    }
    await node.set_variables(v)
    await client.sleep(1)

    for i in range(N):
        right = node['motor.right.speed']
        left = node['motor.left.speed']

        rdata.append(right)
        ldata.append(left)

        vel = (right + left)/2
        vel_error.append(target - vel)
        await client.sleep(T)

    print(f'Velocity error variance: {np.var(vel_error)}')

    actual_pos = 0
    ideal_pos = 0
    pos_error = []
    for i in range(len(vel_error)):
        ideal_pos += T*target
        actual_pos += T*(rdata[i]+ldata[i])/2
        pos_error.append([ideal_pos - actual_pos])

    print(pos_error)
    print(f'Positional error variance: {np.var(pos_error)}')

    plt.subplot(2, 1, 1)
    plt.hist(vel_error)
    plt.title("Velocity error")
    plt.subplot(2, 1, 2)
    plt.hist(pos_error)
    plt.title("Positional error")
    plt.show()

async def test_angular_variances():
    global node
    await init_tdm()

    N = 500
    target_l = 160
    target_r = 100
    target_w = (target_l - target_r)/constants.robot['wheel_base']
    T = 0.05

    vel_error = []
    rdata = []
    ldata = []

    v = {
        "motor.left.target": [target_l*constants.robot['linear_to_mtr_val']],
        "motor.right.target": [target_r*constants.robot['linear_to_mtr_val']],
    }
    await node.set_variables(v)
    await client.sleep(1)

    for i in range(N):
        right = node['motor.right.speed']
        left = node['motor.left.speed']

        rdata.append(right)
        ldata.append(left)

        w = (left - right)/constants.robot['wheel_base']
        vel_error.append(target_w - w)
        await client.sleep(T)

    print(f'Angular velocity error variance: {np.var(vel_error)}')

    actual_pos = 0
    ideal_pos = 0
    pos_error = []
    for i in range(len(vel_error)):
        ideal_pos += T*target_w
        actual_pos += T*(ldata[i] - rdata[i])/constants.robot['wheel_base']
        pos_error.append([ideal_pos - actual_pos])

    print(pos_error)
    print(f'Angle error variance: {np.var(pos_error)}')

    plt.subplot(2, 1, 1)
    plt.hist(vel_error)
    plt.title("Angular velocity error")
    plt.subplot(2, 1, 2)
    plt.hist(pos_error)
    plt.title("Angle error")
    plt.show()



def variables_changed(node, variables):
    try:
        pass
    except KeyError:
        pass
    

# Setup tdm client
async def init_tdm():
    global node, state
    node = await client.wait_for_node()
    await node.lock()
    await node.wait_for_variables()
    await node.watch(variables=True)
    node.add_variables_changed_listener(variables_changed)
    print("Initialized")

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
    

if __name__ == "__main__":
    print("Hello, World!")
    client = ClientAsync()
    client.run_async_program(test_velocity_to_motor_input)
    client.close()
    print("stopped")

    os._exit(0)