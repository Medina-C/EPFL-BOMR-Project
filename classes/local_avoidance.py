# avoidance_edge_follow.py
"""
Edge-following obstacle avoidance using horizontal proximity sensors only.

Assumptions:
- Obstacles are box-like.
- We only have front-facing proximity sensors (no side sensors).
- We use "edge following" (like wall-following) where the robot keeps the obstacle
  on its chosen side until it no longer senses it.
- High sensitivity thresholds are used (user choice).

How it works (high level):
1. GO_TO_GOAL: robot drives forward continuously.
2. If front sensors detect an obstacle -> decide which side to follow (left or right).
3. FOLLOW_EDGE: maintain a target side-distance using a proportional controller.
   - If following left: use prox[1] (left) primarily, prox[2]/prox[3] as supplementary.
   - If following right: use prox[5] (right) primarily, prox[4]/prox[3] as supplementary.
4. Exit FOLLOW_EDGE when the chosen side sensor reads "clear" for several checks.
5. Re-enter GO_TO_GOAL and continue.

Usage:
    python avoidance_edge_follow.py

Notes:
- Tweak PROPORTIONAL_GAIN and TARGET_SIDE_VALUE to tune following behaviour.
- Timing and thresholds chosen for "high sensitivity" (detect early).
"""

from tdmclient import ClientAsync, aw
import os
import csv
import time

# ---------------------------
# Configuration
# ---------------------------
FRONT_THRESHOLD = 1600   # threshold for backup safety check (lower = triggers earlier when closer)
CLEAR_THRESHOLD = 800   # threshold for exit condition (lower = exits when sensors are lower/clearer)
CHECK_INTERVAL = 0.05    # seconds between loop iterations
FORWARD_SPEED = 160      # forward speed
TURN_SPEED = 160         # turning base speed used in corrections
PROPORTIONAL_GAIN = 0.001  # proportional gain for side error -> motor differential (increased for more reactivity)
TARGET_SIDE_VALUE = 1200  # desired side prox reading while following (higher = maintain closer distance to edge, lower = maintain farther distance)
EXIT_CONSISTENT_COUNT = 5    # number of consecutive "clear" readings to exit

# sensor indices for prox.horizontal (front arc only)
# 0: NO  (front-left outer), 1: NNO (front-left inner),
# 2: N   (front-center),     3: NNE (front-right inner),
# 4: NE  (front-right outer), 5/6 = rear sensors (ignored here)
IDX_NO = 0
IDX_NNO = 1
IDX_FRONT = 2
IDX_NNE = 3
IDX_NE = 4


# ---------------------------
# Helpers: sensors & motors
# ---------------------------



class LocalNav():
    def __init__(self):
        self.clear_count = 0
        self.follow_side = None  # Track which side we're following

    async def follow_edge(self, node, prox):
        """Edge-following with proportional control. Called from main loop."""
        
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
        error = TARGET_SIDE_VALUE - sensor1
        diff = PROPORTIONAL_GAIN * error * TURN_SPEED
        
        # Limit diff to ensure both motors always move forward (prevent in-place turning)
        max_diff = FORWARD_SPEED * 0.8  # Allow up to 80% of forward speed as differential
        diff = clamp(diff, -max_diff, max_diff)

        if self.follow_side == 'left':
            left_speed = FORWARD_SPEED - diff
            right_speed = FORWARD_SPEED + diff
        elif self.follow_side == 'right':
            left_speed = FORWARD_SPEED + diff
            right_speed = FORWARD_SPEED - diff

        left_speed = clamp(left_speed, -500, 500)
        right_speed = clamp(right_speed, -500, 500)

        # SAFETY: Back up if too close
        if front_val > FRONT_THRESHOLD * 1.1:
            await set_motors(node, -120, -120)
        else:
            await set_motors(node, left_speed, right_speed)
            

        # Check for clearance
        if sensor1 < CLEAR_THRESHOLD and sensor2 < CLEAR_THRESHOLD:
            self.clear_count += 1
            #await set_motors(node, 100, 100)
    
        else:
            self.clear_count = 0

        cleared = self.clear_count >= EXIT_CONSISTENT_COUNT
        if cleared:
            # Reset state for next obstacle
            self.follow_side = None
            self.clear_count = 0
        
        return cleared


def read_prox(node):
    """Return 7 horizontal proximity values as a list of ints (safe: returns zeros on error)."""
    try:
        v = node["prox.horizontal"]
        # convert to plain list (sometimes it's a memoryview-like)
        return list(v)
    except Exception:
        return [0] * 7


def clamp(x, a, b):
    return max(a, min(b, x))


async def set_motors(node, left, right):
    """Set motor targets directly (simple, like the line-follow example)."""
    left_cmd = int(clamp(left, -500, 500))
    right_cmd = int(clamp(right, -500, 500))
    
    await node.set_variables({
        "motor.left.target": [left_cmd],
        "motor.right.target": [right_cmd]
    })
