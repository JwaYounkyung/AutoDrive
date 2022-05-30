import cantools
import can
from pprint import pprint
from time import sleep

db = cantools.database.load_file("/home/glad/final_for_class/Santafe_Final.dbc")

def can_read (message):
    de_message = db.decode_message(message.arbitration_id, message.data)
    pprint(de_message)
    return de_message

def main():
    can_bus = can.interface.Bus('vcan0', bustype='socketcan')
    # can_bus = can.interface.pcan.PcanBus()

    while True:
        message = can_bus.recv()
        try:
            can_read(message)
        except KeyError:
            pass


if __name__ == '__main__':
    main()
