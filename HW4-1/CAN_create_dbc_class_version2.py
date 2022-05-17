# pip install canlib
# This file creates CAN DBC file
# CAN 2.0a
# 500 Kpbs
# Little endian
# reference : https://www.fatalerrors.org/a/0N591zs.html & Santafe_Digist_CAN_Definition_ver0.1.xlsx
# https://pycanlib.readthedocs.io/en/v1.17.748/examples/create_db.html


import argparse
from collections import namedtuple
from canlib import kvadblib

Message = namedtuple('Message', 'name id dlc signals')
Signal = namedtuple('Signal', 'name size scaling limits unit sigtype')
EnumSignal = namedtuple('EnumSignal', 'name size scaling limits unit enums sigtype') #Enumeration signal : it is used to define a list of labels

_messages = [
Message(
        name='Control_CMD',
        id=336,
        dlc=8, #Data length count,
        signals=[
            EnumSignal(
                name='Override', #Packet Name
                size=(0,8),          #Signal Byte Number and signal length
                scaling=(1,0),       #Scaling and offset = raw_value*1 + 0
                limits=(0,1),        #vaild real values are in the range 0 to 0
                unit="",             #it will be ignored
                enums={'Overide_On': 0, "Override_Off": 1},
                sigtype=102 #signed 1 unsigned 2 
            ),
            Signal(
                name='Alive_Count',
                size=(8,8),
                scaling=(1,0),
                limits=(0,255),
                unit="",
                sigtype=2
            ),
            Signal(
                name='Angular_Speed_CMD',
                size=(40, 8),
                scaling=(1, 0),
                limits=(0, 255),
                unit="",
                sigtype=2
            )
        ]
    ),
Message(
        name='Driving_CMD',
        id=338,
        dlc=8, 
        signals=[
            Signal(
                name='Accel_CMD', 
                size=(8,16),
                scaling=(1,0),
                limits=(650,3400),
                unit="",
                sigtype=2
            ),
            Signal(
                name='Brake_CMD',
                size=(8,16),
                scaling=(1,0),
                limits=(0,17000),
                unit="",
                sigtype=2
            ),
            Signal(
                name='Steering_CMD',
                size=(40, 16),
                scaling=(1, 0),
                limits=(-520, 520),
                unit="deg",
                sigtype=1
            ),
            EnumSignal(
                name='Gear_Shift_CMD',
                size=(40, 8),
                scaling=(1, 0),
                limits=(0, 7),
                unit="",
                enums={'Parking': 0, "Driving": 5, "Neutral": 6, "Reverse": 7},
                sigtype=102 
            )
        ]
    ),
Message(
        name='Vehicle_Info_1',
        id=81,
        dlc=8, 
        signals=[
            Signal(
                name='APS_Feedback', 
                size=(8,16),
                scaling=(1,0),
                limits=(0,3800),
                unit="",
                sigtype=2
            ),
            Signal(
                name='Brake_ACT_Feedback',
                size=(8,16),
                scaling=(1,0),
                limits=(0,35000),
                unit="",
                sigtype=2
            ),
            EnumSignal(
                name='Gear_Shift_Feed',
                size=(40, 8),
                scaling=(1, 0),
                limits=(0, 7),
                unit="",
                enums={'Parking': 0, "Driving": 5, "Neutral": 6, "Reverse": 7},
                sigtype=102
            ),
            Signal(
                name='Steering_Angle_Feedback',
                size=(40, 16),
                scaling=(1, -0.1),
                limits=(-540, 540),
                unit="",
                sigtype=1
            ),
            Signal(
                name='Switch_State',
                size=(40, 8),
                scaling=(1, 0),
                limits=(0, 255), #### not yet
                unit="",
                sigtype=2
            )
        ]
    ),
Message(
        name='Vehicle_Info_2',
        id=82,
        dlc=8,
        signals=[
            EnumSignal(
                name='Override_Feedback',
                size=(8,8),
                scaling=(1,0),
                limits=(0,6),
                unit="",
                enums={'Manual': 0, "Auto": 1, "Steer": 2, "Accel": 3, "Brake": 4, "Sensor":5, "E_Stop":6},
                sigtype=102
            ),
            Signal(
                name='Vehicle_Speed',
                size=(8,16),
                scaling=(1,0),
                limits=(0,255),
                unit="km/h",
                sigtype=2
            )
        ]
    ),
Message(
        name='Vehicle_Info_3',
        id=83,
        dlc=8, 
        signals=[
            Signal(
                name='RPM', 
                size=(8,16),
                scaling=(1,0),
                limits=(0,255), ### not yet
                unit="",
                sigtype=2
            )
        ]
    ),
Message(
        name='Vehicle_Info_4',
        id=84,
        dlc=8, 
        signals=[
            Signal(
                name='Wheel_Speed_Rear_Right', 
                size=(8,16),
                scaling=(1,0.1),
                limits=(0,255),
                unit="km/h",
                sigtype=2
            ),
            Signal(
                name='Wheel_Speed_Rear_Left', 
                size=(8,16),
                scaling=(1,0.1),
                limits=(0,255),
                unit="km/h",
                sigtype=2
            ),
            Signal(
                name='Wheel_Speed_Rear_Left', 
                size=(8,16),
                scaling=(1,0.1),
                limits=(0,255),
                unit="km/h",
                sigtype=2
            ),
            Signal(
                name='Wheel_Speed_Rear_Right', 
                size=(8,16),
                scaling=(1,0.1),
                limits=(0,255),
                unit="km/h",
                sigtype=2
            )
        ]
    )
]
def create_database(name, filename):
    db = kvadblib.Dbc(name=name)

    for _msg in _messages:
        message = db.new_message(
            name=_msg.name,
            id=_msg.id,
            dlc=_msg.dlc
        )

        for _sig in _msg.signals:
            if isinstance(_sig, EnumSignal):
                #_type = kvadblib.SignalType.ENUM_UNSIGNED
                _enums = _sig.enums
            else:
                #_type = kvadblib.SignalType.UNSIGNED
                _enums = {}
            print(_sig.sigtype)

            message.new_signal(
                name=_sig.name,
                type=kvadblib.SignalType(_sig.sigtype),
                byte_order=kvadblib.SignalByteOrder.INTEL,
                mode=kvadblib.SignalMultiplexMode.MUX_INDEPENDENT,
                size=kvadblib.ValueSize(*_sig.size),
                scaling=kvadblib.ValueScaling(*_sig.scaling),
                limits=kvadblib.ValueLimits(*_sig.limits),
                unit=_sig.unit,
                enums=_enums
            )

    db.write_file(filename)
    db.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create a database from scratch.")
    parser.add_argument('--filename', default="/home/jwa/Desktop/Code/AutoDrive/HW4-1/mydbc.dbc", help=("The filename to save the database to."))
    parser.add_argument(
        '-n',
        '--name',
        default='Santafe Example',
        help=("The name of the database")
    )
    args = parser.parse_args()

    create_database(args.name, args.filename)

    print("The DBC file has been created!")
