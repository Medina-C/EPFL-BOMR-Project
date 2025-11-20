import constants
from ekf import EKF
from motion_control import do_astolfi

from classes.gp import GlobalPlanner, Plotter
from classes.cv import Vision

import numpy as np
from tdmclient import ClientAsync

state = 'global_nav'

# Units: seconds, mm


# Setup tdm client
async def init_tdm():
    global node, state
    node = await client.wait_for_node()
    await node.lock()
    await node.wait_for_variables()
    await node.watch(variables=True)
    node.add_variables_changed_listener(variables_changed)

# Main program
async def main():
    global node, state
    await init_tdm()
    
    # TODO CV and path initialization stuff

    # TODO dummy values to change
    start_pose = (0, 0, 0)
    goal_pose = (300, 300)

    # EKF initialization
    ekf = EKF(
        (constants.ekf['move_std']**2)*np.eye(len(start_pose)), 
        (constants.ekf['measure_std']**2)*np.eye(len(start_pose)),
        'basic',
        start_pose,
        constants.robot['wheel_base'],
        constants.general['dt']
    )

    # Main loop
    while state != 'stop':
        current_pose = get_current_pose(ekf)

        # Conditions
        if reached_goal(current_pose, goal_pose):
            state = 'stop'
            break
        elif obstacle_detected():
            state = 'local_nav'

        # Act based on states
        if state == 'global_nav':
            await global_nav(start_pose, goal_pose)
        elif state == 'local_nav':
            await local_nav()

        await client.sleep(constants.general['dt'])

    # Stop
    await set_motors(0, 0)

# Helper function for setting motor values
# takes LINEAR SPEEDS in mm/s as input
async def set_motors(left, right):
    global node
    await node.set_variables({
        "motor.left.target": [left * constants.robot['linear_to_mtr_val']],
        "motor.right.target": [right * constants.robot['linear_to_mtr_val']],
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
def get_current_pose(ekf: EKF):
    global node
    inputs = np.array(
        node['motor.left.speed'] * constants.robot['mtr_val_to_linear'], 
        node['motor.right.speed'] * constants.robot['mtr_val_to_linear'])
    measurements = ekf.state # TODO use real measurements

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

async def global_nav(start_pose, goal_pose):
    vl, vr = do_astolfi(start_pose, goal_pose)
    await set_motors(vl, vr)
    

async def local_nav():
    pass


if __name__ == "__main__":
    client = ClientAsync()
    client.run_async_program(main)
    client.close()
    print("stopped")
    print(state)