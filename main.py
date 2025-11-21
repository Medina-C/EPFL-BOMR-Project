import constants as constants
from classes.ekf import EKF
from classes.motion_control import do_astolfi_cl

from classes.gp import GlobalPlanner, Plotter
from classes.cv import Vision

import matplotlib.pyplot as plt
import numpy as np
from tdmclient import ClientAsync
from time import time

state = 'global_nav'

# Setup tdm client
async def init_tdm():
    global node, state
    node = await client.wait_for_node()
    await node.lock()
    await node.wait_for_variables()
    await node.watch(variables=True)
    node.add_variables_changed_listener(variables_changed)
    print("Initialized")

# Main program
async def main():
    global node, state
    await init_tdm()
    
    # TODO CV and path initialization stuff

    # TODO dummy values to change
    start_pose = (0, 0, 0, np.pi/2, 0)
    goals = [
        (0, 300, np.pi/2),
        (0, 600, np.pi/2),
        (300, 600, 0),
        (600, 600, 0),
    ]
    g = 0
    goal_pose = goals[g]

    # EKF initialization
    ekf = EKF(
        (constants.ekf['move_std']**2)*np.eye(len(start_pose)), 
        (constants.ekf['measure_std']**2)*np.eye(len(start_pose)),
        'complex',
        start_pose,
        constants.robot['wheel_base'],
        constants.general['dt']
    )
    state_log = [start_pose]

    # Main loop
    while state != 'stop':
        print(state)
        t = time()
        current_pose = get_current_pose(ekf)
        state_log.append(current_pose)
        print(f"1 - {time()-t}")
        # Conditions
        if reached_goal(current_pose, goal_pose):
            g += 1
            if g < len(goals):
                goal_pose = goals[g]
            else:
                print("Reached goal!")
                state = 'stop'
                break
        print(f"2 - {time()-t}")
        # elif obstacle_detected():
        #     state = 'local_nav'

        # Act based on states
        if state == 'global_nav':
            await global_nav(current_pose, goal_pose, t)
            print(f"4 - {time()-t}")
        elif state == 'local_nav':
            await local_nav()
        else:
            print(f"Invalid state {state}!")

        # Processing takes ~100 ms. dt is slightly higher -> wait for the remainder
        sleep_time = constants.general['dt'] - (time()-t)
        print(sleep_time)
        if sleep_time > 0:
            await client.sleep(sleep_time)

    # Stop
    await set_motors(0, 0)

    plt.plot([state[0] for state in state_log], [state[1] for state in state_log])
    plt.axis('equal')
    plt.figure()
    plt.plot([state[3] for state in state_log])
    plt.show()

# Helper function for setting motor values
# takes LINEAR SPEEDS in mm/s as input
async def set_motors(left_vel, right_vel):
    global node

    left_in = int(left_vel * constants.robot['linear_to_mtr_val'])
    right_in = int(right_vel * constants.robot['linear_to_mtr_val'])

    if left_in > 500:
        left_in = 500
    elif left_in < -500:
        left_in = -500
    if right_in > 500:
        right_in = 500
    elif right_in < -500:
        right_in = -500

    print(left_in, right_in)

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
            state = 'stop'
            print("stop!")
    except KeyError:
        pass

# Use the EKF to update the current estimated pose
# TODO replace measurements with CV measurements
def get_current_pose(ekf: EKF):
    global node
    vl = node['motor.left.speed']
    vr = node['motor.right.speed']
    inputs = np.array([
        vl * constants.robot['mtr_val_to_linear'], 
        vr * constants.robot['mtr_val_to_linear']
    ])

    v = (vl+vr)/constants.robot['wheel_base']
    w = (vr-vl)/constants.robot['wheel_base']

    measurements = [
        ekf.state[0],
        ekf.state[1],
        v,
        ekf.state[3],
        w
    ]

    return ekf.generate_next_state(inputs, measurements)

def reached_goal(current, goal):
    px, py = current[0], current[1]
    gx, gy = goal[0], goal[1]

    # If we're at both coordinates (within tolerance)
    # then we've reached the goal (regardless of angle)
    return (abs(px - gx) < constants.general['goal_buffer'] and
        abs(py - gy) < constants.general['goal_buffer'])

def obstacle_detected():
    global node
    prox = node['prox.horizontal']

    for value in prox:
        if value > 0:
            return True
        
    return False

async def global_nav(start_pose, goal_pose, t):
    vl, vr = do_astolfi_cl(start_pose, goal_pose)
    print(f"3 - {time()-t}")
    await set_motors(vl, vr)
    

async def local_nav():
    pass


if __name__ == "__main__":
    print("Hello, World!")
    client = ClientAsync()
    client.run_async_program(main)
    client.close()
    print("stopped")
    print(state)