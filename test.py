import constants as constants
from constants import * #TODO ugly, change
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

    vision = Vision(0)
     
    while True:
        vision.update_vision(False, False, False)
        if vision.frame_disp is not None:
            cv.imshow('frame', vision.frame_disp)
        key = cv.waitKey(1) & 0xFF
        if key == 27:  # ESC -> exit
            break
        if key == ord('h'):
            vision.update_vision(False, True, True) 

    output = open('outputs/speed_to_mtr_input.csv', 'a')

    delay = 10

    for speed in [100, 150, 200, 250]:

        v = {
            "motor.left.target": [speed],
            "motor.right.target": [speed],
        }
        await node.set_variables(v)

        # Wait for acceleration
        await client.sleep(1)
        vision.update_vision(False, False, False)
        vision.get_tag_pose(4)
        start_pose = vision.thymio_pose['pose_mm'].flatten()
        print(start_pose)

        t = time()
        while time()-t < delay:
            vision.update_vision(False, False, False)
            vision.get_tag_pose(4)
            pose = vision.thymio_pose['pose_mm'].flatten()
            print(pose)
            await client.sleep(0.05)

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

        output.write(f"{abs(speed)},{velocity}\n")

    output.close()

async def test_motor_l_r():
    global node
    await init_tdm()

    vision = Vision(0)
     
    while True:
        vision.update_vision(False, False, False)
        if vision.frame_disp is not None:
            cv.imshow('frame', vision.frame_disp)
        key = cv.waitKey(1) & 0xFF
        if key == 27:  # ESC -> exit
            break
        if key == ord('h'):
            vision.update_vision(False, True, True) 

    output = open('outputs/motor_rl_constants2.csv', 'a')

    delay = 10
    for speed in[115, -115, 135, -135, 170, -170, 190, -190]:

        v = {
            "motor.left.target": [int(speed*0.99)],
            "motor.right.target": [speed],
        }
        await node.set_variables(v)

        # Wait for acceleration
        await client.sleep(1)
        vision.update_vision(False, False, False)
        vision.get_tag_pose(4)
        start_th = vision.thymio_pose['theta']
        print(f'start={start_th}')
        t = time()
        while time()-t < delay:
            vision.update_vision(False, False, False)
            vision.get_tag_pose(4)
            pose = vision.thymio_pose['theta']
            await client.sleep(0.05)

      

        end_th = vision.thymio_pose['theta']
        print(f'end={end_th}')

        v = {
            "motor.left.target": [0],
            "motor.right.target": [0],
        }
        await node.set_variables(v)
        await client.sleep(1)

        theta_diff = end_th - start_th
        if speed>0:
            output.write(f"{abs(speed)},{theta_diff}\n")

    output.close()

async def test_linear_variances_allcv():
    global node
    await init_tdm()

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

    target_mm = 100
    target = target_mm*constants.robot['linear_to_mtr_val']

    v = {
        "motor.left.target": [int(target*0.99)],
        "motor.right.target": [int(target)],
    }
    await node.set_variables(v)
    await client.sleep(1)

    delay = 5

    pos_log = [0]
    vel_error = []
    pos_error = []
    rdata = []
    ldata = []
    time_log = [time()]

    vision.get_tag_pose(4)
    start_pose = vision.thymio_pose['pose_mm'].flatten()

    while time_log[-1] - time_log[0] < delay:
        vision.update_vision(False, False, False)
        vision.get_tag_pose(4)
        time_log.append(time())
        vel_cv = vision.get_tag_velocity(4)['speed_mm_s']

        right = node['motor.right.speed']*constants.robot['mtr_val_to_linear']
        left = node['motor.left.speed']*constants.robot['mtr_val_to_linear']

        rdata.append(right)
        ldata.append(left)

        vel_robot = (right + left)/2
        vel_error.append(vel_cv - vel_robot)

        print(f'Robot vel: {vel_robot}, CV vel: {vel_cv}, Error: {vel_cv - vel_robot}')

        dt = time_log[-1] - time_log[-2]
        pos_robot = pos_log[-1] + vel_robot*dt

        dx = start_pose[0] - vision.thymio_pose['pose_mm'].flatten()[0]
        dy = start_pose[1] - vision.thymio_pose['pose_mm'].flatten()[1]

        pos_cv = np.sqrt(dx*2 + dy*2)
        pos_log.append(pos_cv)
        pos_error.append(pos_cv - pos_robot)

        print(f'Robot pos: {pos_robot}, CV pos: {pos_cv}, Error: {pos_cv - pos_robot}')
        await client.sleep(0.05)
    
    v = {
        "motor.left.target": [0],
        "motor.right.target": [0],
    }
    await node.set_variables(v)
    await client.sleep(1)

    print(f'Velocity error variance: {np.var(vel_error)}')
    print(f'Positional error variance: {np.var(pos_error)}')

    plt.subplot(2, 1, 1)
    plt.hist(vel_error)
    plt.title("Velocity error")
    plt.subplot(2, 1, 2)
    plt.hist(pos_error)
    plt.title("Positional error")
    plt.show()

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
    vision.get_tag_pose(4)

    target_mm = 100
    target_robot = target_mm*constants.robot['linear_to_mtr_val']

    v = {
        "motor.left.target": [int(target_robot*0.99)],
        "motor.right.target": [int(target_robot)],
    }
    await node.set_variables(v)
    await client.sleep(1)

    vel_error = []
    pos_error = []
    rdata = []
    ldata = []

    delay = 8

    time_log = [time()]
    vision.update_vision(False, False, False)
    vision.get_tag_pose(4)

    start_pose = vision.thymio_pose['pose_mm'].flatten()
    prev_cv_pos = 0

    output = open('outputs/robot_errors.csv', 'a')

    await client.sleep(0.1)

    while time_log[-1] - time_log[0] < delay:
        right = node['motor.right.speed']
        left = node['motor.left.speed']
        time_log.append(time())

        rdata.append(right)
        ldata.append(left)

        vel = (right + left)/2*constants.robot['mtr_val_to_linear']
        vel_error.append(target_mm - vel)

        vision.update_vision(False, False, False)
        vision.get_tag_pose(4)


        dx = start_pose[0] - vision.thymio_pose['pose_mm'].flatten()[0]
        dy = start_pose[1] - vision.thymio_pose['pose_mm'].flatten()[1]

        pos_cv = np.sqrt(dx**2 + dy**2)
        pos_robot = prev_cv_pos + vel*(time_log[-1] - time_log[-2])
        print(pos_cv, pos_robot, vel, time_log[-1] - time_log[-2])

        # Skip first one because something weird is happening with the get_tag_pose() timing
        if prev_cv_pos != 0:
            pos_error.append(pos_cv - pos_robot)

            output.write(f'{vel_error[-1]},{pos_error[-1]}\n')

        prev_cv_pos = pos_cv

        await client.sleep(0.1)

    v = {
        "motor.left.target": [0],
        "motor.right.target": [0],
    }
    await node.set_variables(v)
    output.close()

    print(f'Velocity error variance: {np.var(vel_error)}')

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
    vision.get_tag_pose(4)

    target_l = 160
    target_r = 90
    target_w = (target_l - target_r)/constants.robot['wheel_base']

    v = {
        "motor.left.target": [int(target_l*0.99*constants.robot['linear_to_mtr_val'])],
        "motor.right.target": [int(target_r*constants.robot['linear_to_mtr_val'])],
    }
    await node.set_variables(v)
    await client.sleep(1)

    w_error = []
    theta_error = []
    rdata = []
    ldata = []

    delay = 30

    time_log = [time()]
    vision.update_vision(False, False, False)
    vision.get_tag_pose(4)

    prev_cv_theta = vision.thymio_pose['theta']

    output = open('outputs/robot_errors.csv', 'a')

    await client.sleep(0.1)

    while time_log[-1] - time_log[0] < delay:
        right = node['motor.right.speed']
        left = node['motor.left.speed']
        time_log.append(time())

        rdata.append(right)
        ldata.append(left)

        w = (left - right)/constants.robot['wheel_base']*constants.robot['mtr_val_to_linear']
        w_error.append(target_w - w)

        vision.update_vision(False, False, False)
        vision.get_tag_pose(4)

        theta_cv = vision.thymio_pose['theta']

        theta_robot = prev_cv_theta + w*(time_log[-1] - time_log[-2])
        print(theta_cv, theta_robot, w, time_log[-1] - time_log[-2])

        # Skip first one because something weird is happening with the get_tag_pose() timing
        if prev_cv_theta != 0:
            theta_error.append(theta_cv - theta_robot)
            output.write(f'{w_error[-1]},{theta_error[-1]}\n')

        prev_cv_theta = theta_cv

        await client.sleep(0.1)

    v = {
        "motor.left.target": [0],
        "motor.right.target": [0],
    }
    await node.set_variables(v)
    output.close()

    print(f'Velocity error variance: {np.var(w_error)}')

    print(f'Positional error variance: {np.var(theta_error)}')

    plt.subplot(2, 1, 1)
    plt.hist(w_error)
    plt.title("Velocity error")
    plt.subplot(2, 1, 2)
    plt.hist(theta_error)
    plt.title("Positional error")
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
    client.run_async_program(test_angular_variances)
    client.close()
    print("stopped")

    os._exit(0)