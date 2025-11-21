from tdmclient import ClientAsync
client = ClientAsync()

WHITE = 900
BLACK = 130
WHITE_THRESH = (WHITE - BLACK)*0.7 + BLACK
BLACK_THRESH = (WHITE - BLACK)*0.3 + BLACK

state = 'follow'

async def follow_line():
    global node, state
    prox = node['prox.horizontal'][0]
    if prox > 4000:
        print("wall")
        state = 'wall'
        return

    light = node['prox.ground.reflected'][0]
    left = 0
    right = 0

    if light > WHITE_THRESH:
        left = 0
        right = 200
    elif light < BLACK_THRESH:
        left = 200
        right = 0
    else:
        left = 100
        right = 100
    await set_motors(left, right)

async def follow_wall():
    global node, state

    light = node['prox.ground.reflected']
    if light[0] < BLACK_THRESH and light[1] < BLACK_THRESH:
        state = 'straight'
        print("straight")
        return

    prox = node['prox.horizontal'][0]
    if prox > 4000:
        await set_motors(150, 50)
    else:
        await set_motors(50, 150)


async def move_straight():
    global node, state

    light = node['prox.ground.reflected']
    if light[0] > BLACK_THRESH or light[1] > BLACK_THRESH:
        state = 'follow'
        print("follow")
        return
    
    await set_motors(100, 100)

async def set_motors(left, right):
    global node
    await node.set_variables({
        "motor.left.target": [left],
        "motor.right.target": [right],
    })

def variables_changed(node, variables):
    global state
    try:
        stop = variables["button.center"]
        if stop:
            state = 'stop'
            print("stop!")
    except KeyError:
        pass

async def prog():
    global node, state
    node = await client.wait_for_node()
    await node.lock()
    await node.wait_for_variables()
    await node.watch(variables=True)
    node.add_variables_changed_listener(variables_changed)
    print("going")

    while state != 'stop':
        if state == 'follow':
            await follow_line()
        elif state == 'wall':
            await follow_wall()
        elif state == 'straight':
            await move_straight()
        else:
            print("error state " + state)
            break
        await client.sleep(0.01)

    await set_motors(0, 0)

client.run_async_program(prog)
client.close()
print("stopped")
print(state)