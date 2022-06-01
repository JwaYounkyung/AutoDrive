import cantools
import can
from pprint import pprint
from time import sleep
from pynput import keyboard

db = cantools.database.load_file("mydbc.dbc")
can_bus = can.interface.Bus('vcan0', bustype='socketcan')

heartbeat = 0
accel = 650
brake = 0
gear = 0
steer = 0
reserve = 0

def Ctrl_CMD (override, heartbeat):
    ctrl_message = db.get_message_by_name('Control_CMD')
    ctrl_data = ctrl_message.encode({'Override':override, 'Alive_Count': heartbeat, 'Angular_Speed_CMD':100})
    ctrl_message_send = can.Message(arbitration_id=ctrl_message.frame_id, data=ctrl_data)
    can_bus.send(ctrl_message_send,timeout=0.001)
    sleep(0.02)

def Drv_CMD(accel, brake, steer, gear, reserve):
    ctrl_message = db.get_message_by_name('Driving_CMD')
    ctrl_data = ctrl_message.encode({'Accel_CMD':accel, 'Brake_CMD': brake, 'Steering_CMD':steer, 'Gear_Shift_CMD':gear})
    ctrl_message_send = can.Message(arbitration_id=ctrl_message.frame_id, data=ctrl_data)
    can_bus.send(ctrl_message_send,timeout=0.001)
    # sleep(0.02)

def on_press(key):
    global heartbeat, accel, brake, steer, gear

    try:
        if key.char == 'w':
            if accel == 650 and brake != 8000:
                accel = 950
                Drv_CMD(accel, brake, steer, gear, reserve)
        elif key.char == 's':
            if accel > 650:
                accel = 650
                Drv_CMD(accel, brake, steer, gear, reserve)
        elif key.char == 'e':
            if brake == 0 and accel != 950:
                brake = 8000
                Drv_CMD(accel, brake, steer, gear, reserve)
        elif key.char == 'd':
            if brake > 0:
                brake = 0
                Drv_CMD(accel, brake, steer, gear, reserve)
        elif key.char == '0' or key.char == '5' or key.char == '6' or key.char == '7':
            if gear != int(key.char) and accel != 950 and brake == 8000:
                gear = int(key.char)
                Drv_CMD(accel, brake, steer, gear, reserve)
        elif key.char == 'z':
            if steer < 520:
                steer += 1
                Drv_CMD(accel, brake, steer, gear, reserve)
        elif key.char == 'x':
            if steer > -520:
                steer -= 1
                Drv_CMD(accel, brake, steer, gear, reserve)    
    except AttributeError:
        print(key)
    
listener = keyboard.Listener(on_press=on_press)
listener.start()

while True:
    if heartbeat < 255:
            Ctrl_CMD(1, heartbeat)
            heartbeat += 1
    else:
            Ctrl_CMD(1, heartbeat)
            heartbeat = 0
    
    