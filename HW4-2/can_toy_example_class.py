import cantools
import can
import keyboard
from pprint import pprint
from time import sleep
db = cantools.database.load_file("/home/glad/final_for_class/Santafe_Final.dbc")
can_bus = can.interface.Bus('vcan0', bustype='socketcan')

def Ctrl_CMD (override, heartbeat):
    ctrl_message = db.get_message_by_name('Control_CMD')
    ctrl_data = ctrl_message.encode({'Override':override, 'Alive_Count': heartbeat, 'Angular_Speed_CMD':100})
    ctrl_message_send = can.Message(arbitration_id=ctrl_message.frame_id, data=ctrl_data,extended_id=False)
    can_bus.send(ctrl_message_send,timeout=0.001)
    sleep(0.02)
def main():
    heartbeat = 0
    accel = 650
    brake = 0
    gear = 0
    steer = 0
    reserve=0
    while True:
        if heartbeat < 255:
             Ctrl_CMD(1, heartbeat)
             heartbeat += 1
        else:
             Ctrl_CMD(1, heartbeat)
             heartbeat = 0




if __name__ == '__main__':
    main()
