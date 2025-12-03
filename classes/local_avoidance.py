from tdmclient import ClientAsync, aw
import os
import csv
import time
import constants
from constants import IDX_NO, IDX_NNO, IDX_FRONT, IDX_NNE, IDX_NE


class LocalNav():
    def __init__(self):
        self.clear_count = 0
        self.follow_side = None  # Track which side we're following

    async def follow_edge(self, node, prox):
        left_outer = prox[IDX_NO]
        left_inner = prox[IDX_NNO]
        left_val = max(left_outer, left_inner)
        front_val = prox[IDX_FRONT]
        front_left_val = left_inner

        right_inner = prox[IDX_NNE]
        right_outer = prox[IDX_NE]
        right_val = max(right_inner, right_outer)
        front_right_val = right_inner

        # Decide which side to follow (only once at the start)
        if self.follow_side is None:
            left_score = left_val + prox[IDX_NNO] * 0.5
            right_score = right_val + prox[IDX_NNE] * 0.5
            if left_score >= right_score:
                self.follow_side = "left"
            else:
                self.follow_side = "right"

        # FOLLOWING PHASE: Use proportional control to maintain distance along the edge
        if self.follow_side == "left":
            sensor1 = left_val
            sensor2 = front_left_val
        else:
            sensor1 = right_val
            sensor2 = front_right_val

        # error: positive means too far, negative means too close
        error = constants.local_avoidance['target_side_value'] - sensor1
        diff = constants.local_avoidance['proportional_gain'] * error * constants.local_avoidance['turn_speed']
        
        # Limit diff to ensure both motors always move forward (prevent in-place turning)
        max_diff = constants.local_avoidance['forward_speed'] * 0.8  # Allow up to 80% of forward speed as differential
        diff = clamp(diff, -max_diff, max_diff)

        if self.follow_side == 'left':
            left_speed = constants.local_avoidance['forward_speed'] - diff
            right_speed = constants.local_avoidance['forward_speed'] + diff
        elif self.follow_side == 'right':
            left_speed = constants.local_avoidance['forward_speed'] + diff
            right_speed = constants.local_avoidance['forward_speed'] - diff

        left_speed = clamp(left_speed, -500, 500)
        right_speed = clamp(right_speed, -500, 500)

        # SAFETY: Back up if too close
        if front_val > constants.local_avoidance['front_threshold'] * 1.1:
            await set_motors(node, -120, -120)
        else:
            await set_motors(node, left_speed, right_speed)
            

        # Check for clearance
        if sensor1 < constants.local_avoidance['clear_threshold'] and sensor2 < constants.local_avoidance['clear_threshold']:
            self.clear_count += 1
           
    
        else:
            self.clear_count = 0

        cleared = self.clear_count >= constants.local_avoidance['exit_consistent_count']
        if cleared:
            # Reset state for next obstacle
            self.follow_side = None
            self.clear_count = 0
        
        return cleared


def read_prox(node):
    try:
        v = node["prox.horizontal"]
        # convert to plain list (sometimes it's a memoryview-like)
        return list(v)
    except Exception:
        return [0] * 7


def clamp(x, a, b):
    return max(a, min(b, x))


async def set_motors(node, left, right):
    left_cmd = int(clamp(left, -500, 500))
    right_cmd = int(clamp(right, -500, 500))
    
    await node.set_variables({
        "motor.left.target": [left_cmd],
        "motor.right.target": [right_cmd]
    })
