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
import asyncio
import os
import csv
import time

# ---------------------------
# Configuration
# ---------------------------
FRONT_THRESHOLD = 1300   # detect obstacle early (increased for earlier detection of spontaneous obstacles)
SIDE_THRESHOLD = 1000
CLEAR_THRESHOLD = 800
CHECK_INTERVAL = 0.05    # seconds between loop iterations
FORWARD_SPEED = 160      # forward speed
TURN_SPEED = 160         # turning base speed used in corrections
PROPORTIONAL_GAIN = 0.0005  # proportional gain for side error -> motor differential (reduced to minimize turning)
TARGET_SIDE_VALUE = 1000  # desired side prox reading while following (tunable)
EXIT_CONSISTENT_COUNT = 6  # number of consecutive "clear" readings to exit

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


# Global CSV logging
csv_file = None
csv_writer = None
start_time = None


async def set_motors(node, left, right):
    """Set motor targets directly (simple, like the line-follow example)."""
    left_cmd = int(clamp(left, -500, 500))
    right_cmd = int(clamp(right, -500, 500))
    
    await node.set_variables({
        "motor.left.target": [left_cmd],
        "motor.right.target": [right_cmd]
    })


async def stop(node):
    await set_motors(node, 0, 0)


# ---------------------------
# State machine states
# ---------------------------
STATE_GO_TO_GOAL = "GO_TO_GOAL"
STATE_FOLLOW_LEFT = "FOLLOW_LEFT"
STATE_FOLLOW_RIGHT = "FOLLOW_RIGHT"

# ---------------------------
# Edge following controllers
# ---------------------------
async def follow_left_edge(node, client, run_until_cleared=True):
    """
    Follow the left edge using a proportional controller.
    Run until the left side is consistently clear (side sensor < CLEAR_THRESHOLD)
    or until external stop (max iterations).
    """
    clear_count = 0
    iteration = 0
    max_iter = 2000  # safety cap (very large)

    while iteration < max_iter:
        client.process_waiting_messages()
        if node["button.center"] == 1:
            print("[FOLLOW_LEFT] Top button pressed – exiting")
            await stop(node)
            os._exit(0)

        prox = read_prox(node)
        # Use the stronger of the two left-facing sensors
        left_outer = prox[IDX_NO]
        left_inner = prox[IDX_NNO]
        left_val = max(left_outer, left_inner)
        front_val = prox[IDX_FRONT]
        front_left_val = left_inner

        # compute side error (positive means too far from wall, negative means too close)
        error = TARGET_SIDE_VALUE - left_val

        # proportional correction: adjust differential between left/right
        diff = PROPORTIONAL_GAIN * error * TURN_SPEED  # scaled diff
        # base forward speed
        left_speed = FORWARD_SPEED - diff
        right_speed = FORWARD_SPEED + diff

        # clamp speeds
        left_speed = clamp(left_speed, -500, 500)
        right_speed = clamp(right_speed, -500, 500)

        await set_motors(node, left_speed, right_speed)

        # exit condition: if left side is clear below CLEAR_THRESHOLD for several checks
        if left_val < CLEAR_THRESHOLD and front_left_val < CLEAR_THRESHOLD:
            clear_count += 1
        else:
            clear_count = 0

        # Additionally, if front becomes very close, do a small reactive step:
        if front_val > FRONT_THRESHOLD * 1.4:
            await set_motors(node, -100, -100)
            await client.sleep(0.12)
            await set_motors(node, left_speed, right_speed)

        await client.sleep(CHECK_INTERVAL)
        iteration += 1

        if run_until_cleared and clear_count >= EXIT_CONSISTENT_COUNT:
            break

    return


async def follow_right_edge(node, client, run_until_cleared=True):
    """Mirror of follow_left_edge for right side."""
    clear_count = 0
    iteration = 0
    max_iter = 2000

    while iteration < max_iter:
        client.process_waiting_messages()
        if node["button.center"] == 1:
            print("[FOLLOW_RIGHT] Top button pressed – exiting")
            await stop(node)
            os._exit(0)

        prox = read_prox(node)
        right_inner = prox[IDX_NNE]
        right_outer = prox[IDX_NE]
        right_val = max(right_inner, right_outer)
        front_val = prox[IDX_FRONT]
        front_right_val = right_inner

        # error: positive means too far, negative means too close
        error = TARGET_SIDE_VALUE - right_val
        diff = PROPORTIONAL_GAIN * error * TURN_SPEED

        left_speed = FORWARD_SPEED + diff
        right_speed = FORWARD_SPEED - diff

        left_speed = clamp(left_speed, -500, 500)
        right_speed = clamp(right_speed, -500, 500)

        await set_motors(node, left_speed, right_speed)

        if right_val < CLEAR_THRESHOLD and front_right_val < CLEAR_THRESHOLD:
            clear_count += 1
        else:
            clear_count = 0

        if front_val > FRONT_THRESHOLD * 1.4:
            await set_motors(node, -100, -100)
            await client.sleep(0.12)
            await set_motors(node, left_speed, right_speed)

        await client.sleep(CHECK_INTERVAL)
        iteration += 1

        if run_until_cleared and clear_count >= EXIT_CONSISTENT_COUNT:
            break

    return


# ---------------------------
# Decision helpers
# ---------------------------
def decide_follow_side(prox):
    """
    Decide whether to follow left or right edge when obstacle detected.
    Strategy:
      - If left-side readings are higher than right-side -> follow left (object closer on left)
      - If right-side readings higher -> follow right
      - If similar -> prefer side with larger front-side reading (front_left vs front_right)
    """
    left_val = max(prox[IDX_NO], prox[IDX_NNO])
    right_val = max(prox[IDX_NNE], prox[IDX_NE])
    left_score = left_val + prox[IDX_NNO] * 0.5
    right_score = right_val + prox[IDX_NNE] * 0.5

    if left_score >= right_score:
        return "left"
    else:
        return "right"


# ---------------------------
# Main continuous loop
# ---------------------------
async def main_loop(node, client, max_iterations=10000):
    """
    Continuous loop that keeps moving toward the goal and switches to edge following.
    Simple pattern: move forward continuously, detect obstacle, follow edge, resume forward.
    """
    state = STATE_GO_TO_GOAL
    iteration = 0

    print("[MAIN] Starting main avoidance loop (edge-following strategy)")
    print("[MAIN] Sensitivity: FRONT=", FRONT_THRESHOLD, " SIDE=", SIDE_THRESHOLD)

    while iteration < max_iterations:
        client.process_waiting_messages()

        await set_motors(node, FORWARD_SPEED, FORWARD_SPEED)

        prox = read_prox(node)
        front = prox[IDX_FRONT]
        left = max(prox[IDX_NO], prox[IDX_NNO])
        right = max(prox[IDX_NNE], prox[IDX_NE])
        front_left = prox[IDX_NNO]
        front_right = prox[IDX_NNE]

        # State machine
        if state == STATE_GO_TO_GOAL:
            # Move forward continuously (like the line-follow example)
            await set_motors(node, FORWARD_SPEED, FORWARD_SPEED)

            # If front detects obstacle -> choose side and switch state
            if front >= FRONT_THRESHOLD or front_left >= FRONT_THRESHOLD or front_right >= FRONT_THRESHOLD:
                side = decide_follow_side(prox)
                print(f"[MAIN] Obstacle detected. Deciding to follow {side} edge.")
                if side == "left":
                    state = STATE_FOLLOW_LEFT
                else:
                    state = STATE_FOLLOW_RIGHT

        elif state == STATE_FOLLOW_LEFT:
            print("[MAIN] Enter FOLLOW_LEFT state")
            # Call edge following: this function loops until the side is cleared
            await follow_left_edge(node, client, run_until_cleared=True)
            # Once cleared, go back to goal
            print("[MAIN] Left edge cleared, resuming GO_TO_GOAL")
            state = STATE_GO_TO_GOAL

        elif state == STATE_FOLLOW_RIGHT:
            print("[MAIN] Enter FOLLOW_RIGHT state")
            await follow_right_edge(node, client, run_until_cleared=True)
            print("[MAIN] Right edge cleared, resuming GO_TO_GOAL")
            state = STATE_GO_TO_GOAL

        # Exit on top button
        if node["button.center"] == 1:
            print("[MAIN] Top button pressed - exiting loop")
            break

        #small sleep to avoid burning CPU, checks done frequently
        await client.sleep(CHECK_INTERVAL)
        iteration += 1

    print("[MAIN] Stopping")
    await stop(node)
    return


# ---------------------------
# Entry point
# ---------------------------
async def main():
    client = ClientAsync()
    client.process_waiting_messages()
    aw(client.sleep(0.2))

    if not client.nodes:
        print("[ERROR] No Thymio robot found (TDM).")
        os._exit(0)

    node = client.nodes[0]
    print(f"[CONNECTED] Thymio ID={node.id}")

    # Lock for exclusive control
    aw(node.lock())
    print("[LOCKED] Node locked")

    # Wait for required variables (one-time wait)
    aw(node.wait_for_variables({"motor.left.target", "motor.right.target", "prox.horizontal", "button.center"}))
    print("[READY] Motors, prox.horizontal, and button available")

    # Initialize CSV logging
    global csv_file, csv_writer, start_time
    csv_file = open("motor_log.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["time", "left_motor", "right_motor"])
    start_time = time.time()
    print("[LOG] Logging motor targets to motor_log.csv")

    try:
        await main_loop(node, client)
    except KeyboardInterrupt:
        print("[INTERRUPTED] user")
    finally:
        await stop(node)
        if csv_file is not None:
            csv_file.close()
            print("[LOG] Closed motor_log.csv")
        print("[EXIT] Stopping and exiting")
        os._exit(0)


if __name__ == "__main__":
    asyncio.run(main())
