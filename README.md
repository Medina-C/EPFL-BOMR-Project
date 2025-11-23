# BOMR Project - Mobile Robotics Course

This project implements autonomous navigation for a Thymio robot in a static 2m × 2m environment. The system uses an external camera for global navigation and Thymio sensors for local navigation.

## Project Structure

### `test_motors.py`
A simple test script to verify basic robot control capabilities. This script tests:
- Connection to Thymio via tdmclient
- Motor control (forward movement and turning)
- Sensor reading (proximity sensors)

## How to Run the Test Script

1. **Prerequisites:**
   - Thymio robot is powered on and connected (via USB or wireless)
   - Thymio Suite is running
   - Conda environment is activated with tdmclient installed

2. **Run the script:**
   ```bash
   python test_motors.py
   ```

3. **What to expect:**
   - The robot will drive forward for 2 seconds
   - Stop briefly
   - Turn right for 1 second
   - Stop
   - Print proximity sensor values to the console

## Understanding the Code

### Why Node Locking is Critical

**The Problem:**
- Thymio Suite (or other programs) can send motor commands to the robot
- Without locking, your commands might be overridden immediately
- This is why the robot appears to receive commands but doesn't move

**The Solution:**
```python
lock_context = aw(node.lock())  # Acquire exclusive control
with lock_context:               # Enter lock context
    # Now Thymio Suite cannot override our commands
    node.set_variables({"motor.left.target": 200, ...})
    node.flush()
```

**What locking does:**
- Tells TDM (Thymio Device Manager) that THIS program has exclusive control
- Prevents Thymio Suite from sending conflicting motor commands
- Ensures your commands are actually executed by the robot
- Lock is automatically released when exiting the `with` block

**Without locking:** Commands are sent but immediately overridden → robot doesn't move  
**With locking:** Commands are sent and executed → robot moves as expected

### How `aw()` Works

The `aw()` function is a helper that converts async operations to synchronous code:

```python
# Without aw(): This is async code (won't work in regular function)
await node.lock()  # ❌ Can't use await in non-async function

# With aw(): Converts async to sync
lock_context = aw(node.lock())  # ✅ Works in regular function
```

**What `aw()` does:**
- Takes an async generator/coroutine
- Runs it in an internal event loop
- Waits for it to complete
- Returns the result synchronously

**Why we need it:**
- Our `test_motors()` function is NOT async (it's a regular `def`)
- But `node.lock()` returns an async generator
- `aw()` bridges the gap between sync and async code
- This is the pattern recommended by tdmclient for simple scripts

**Example:**
```python
# These are equivalent:
# Async version (if function was async):
async def test():
    async with node.lock():
        ...

# Sync version with aw():
def test():
    with aw(node.lock()):
        ...
```

### Why Thymio Suite Might Override Commands

**Thymio Suite's behavior:**
- Thymio Suite continuously monitors the robot
- It may send default motor commands (speed = 0) if no program is controlling it
- These default commands can override your commands if you don't have exclusive lock

**Common scenarios:**
1. **No lock:** Suite sends default commands → your commands are ignored
2. **Lock acquired:** Your program has control → Suite stops sending commands
3. **Lock released:** Suite resumes control → robot stops moving

**How to verify:**
- Check the lock status in debug output
- If lock fails, the robot won't move
- Always ensure lock is acquired before sending motor commands

### Async Logic Explanation

The script uses **asynchronous programming** patterns via the `aw()` helper because:
- Robot communication operations (sending commands, reading sensors) can take time
- Async allows the program to wait for operations without blocking
- This is the standard pattern for tdmclient

**Key concepts:**
- `aw()` - helper function that runs async code synchronously
- `node.lock()` - returns an async generator for exclusive control
- `node.set_variables()` - sets robot variables (synchronous)
- `node.flush()` - sends pending commands (synchronous)
- `client.sleep()` - async sleep function (use with `aw()`)

**Flow:**
```
Main program (sync function)
  └─> ClientAsync()  [creates client connection]
       └─> client.process_waiting_messages()  [discovers robots]
            └─> aw(node.lock())  [acquires exclusive lock]
                 └─> with lock_context:  [enters lock]
                      └─> node.set_variables()  [sets motor speeds]
                      └─> node.flush()  [sends commands]
                      └─> aw(client.sleep(2))  [waits 2 seconds]
                      └─> [lock released automatically]
```

### Motor Commands Explanation

Motor control in Thymio works by setting target speeds for the left and right motors:

- **Motor speed range:** -500 to 500
- **Positive values:** Motor rotates forward
- **Negative values:** Motor rotates backward
- **Zero:** Motor stops

**Movement patterns:**
- **Forward:** Both motors positive (e.g., left=200, right=200)
- **Backward:** Both motors negative (e.g., left=-200, right=-200)
- **Turn right:** Left motor positive, right motor negative (e.g., left=200, right=-200)
- **Turn left:** Left motor negative, right motor positive (e.g., left=-200, right=200)

**How commands are sent:**
```python
await node.set_variables({
    "motor.left.target": 200,   # Set left motor speed
    "motor.right.target": 200   # Set right motor speed
})
```

The `set_variables()` function sends these values to the robot. The robot will continue moving at these speeds until you send new commands or set them to zero.

### Test Sequence Breakdown

1. **Connect to Thymio:**
   - `ClientAsync(zeroconf=True)` creates a client and connects to TDM
   - `client.process_waiting_messages()` discovers available robots
   - Gets the first robot node: `node = client.nodes[0]`

2. **Wait for Motor Variables:**
   - `aw(node.wait_for_variables({...}))` ensures motor variables are available
   - This prevents errors when trying to set motor speeds

3. **Lock Node (CRITICAL STEP):**
   - `lock_context = aw(node.lock())` acquires exclusive control
   - This prevents Thymio Suite from overriding commands
   - Enter lock context: `with lock_context:`

4. **Drive Forward (2 seconds):**
   - Sets both motors to 200: `node.set_variables({"motor.left.target": 200, ...})`
   - `node.flush()` immediately sends commands to robot
   - `client.process_waiting_messages()` ensures commands are processed
   - `aw(client.sleep(2))` waits 2 seconds while robot moves

5. **Stop:**
   - Sets both motors to 0
   - Flushes and processes messages
   - Brief pause to ensure robot stops

6. **Turn Right (1 second):**
   - Left motor: 200 (forward), Right motor: -200 (backward)
   - This differential drive causes right turn
   - Waits 1 second

7. **Final Stop:**
   - Sets both motors to 0
   - Ensures robot is stopped

8. **Read Sensors:**
   - `aw(node.wait_for_variables({"prox.horizontal"}))` waits for sensor data
   - `node.v.prox.horizontal` accesses the 7 horizontal proximity sensors
   - Returns array of 7 values (0 = no obstacle, higher = closer obstacle)
   - Sensors are arranged left to right on the robot

9. **Release Lock:**
   - Lock is automatically released when exiting the `with` block
   - Thymio Suite can resume control if needed

## Troubleshooting

### Robot Doesn't Move

**Most common cause: Node not locked**
- **Symptom:** Script runs without errors, prints "commands sent", but robot doesn't move
- **Solution:** Check that lock is acquired successfully (look for "✓ Node lock acquired" in output)
- **If lock fails:** Another program might be controlling the robot - close Aseba Studio, Scratch, etc.

**Other causes:**
- Robot is locked by another program (check `python -m tdmclient list` - status should be 2)
- Robot is on unsuitable surface (too dark/reflective - use white/light surface)
- Battery is low (charge the robot)
- Thymio Suite is actively controlling robot (close any active programs in Suite)

**Debug steps:**
1. Run `python -m tdmclient list` - verify robot status is 2 (available)
2. Check script output for "✓ Node lock acquired successfully"
3. Verify "✓ Inside lock context" message appears
4. If lock fails, close all other Thymio programs and try again

### Connection Errors

**"ConnectionRefusedError" when running script:**
- Thymio Suite is not running - start Thymio Suite first
- TDM (Thymio Device Manager) is not active - restart Thymio Suite
- Firewall blocking connection - check Windows Firewall settings

**"No Thymio robot found":**
- Robot is not powered on
- Robot is not connected (check USB/wireless connection)
- TDM hasn't discovered robot yet - wait a few seconds and try again
- Run `python -m tdmclient list` to verify robot is visible

### Import Errors

- Activate your Conda environment: `conda activate tdm_env`
- Verify tdmclient installation: `pip list | grep tdmclient`
- Reinstall if needed: `pip install tdmclient`

### Lock Errors

**"Failed to lock node":**
- Another program has exclusive control
- Close Aseba Studio, Scratch, Blockly, or any other Thymio program
- Wait a few seconds and try again
- Restart Thymio Suite if problem persists

## Next Steps

After confirming this test works, the next task will be implementing local obstacle avoidance behaviors.

