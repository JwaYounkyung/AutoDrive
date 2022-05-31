import cantools
import can
from pprint import pprint

db = cantools.database.load_file("mydbc.dbc")

def can_read (message):
    de_message = db.decode_message(message.arbitration_id, message.data)
    pprint(de_message)
    return de_message

def main():
    can_bus = can.interface.Bus('vcan0', bustype='socketcan')
    # can_bus = can.interface.pcan.PcanBus()

    while True:
        message = can_bus.recv()
        
        if message.arbitration_id == 0x152:
            try:
                can_read(message)
            except:
                pass

if __name__ == '__main__':
    main()