from pynput.keyboard import Key, Controller, Listener
import time

def on_press(key):
    print('{0} pressed'.format(
        key))

def on_release(key):
    print('{0} release'.format(
        key))
    if key == Key.esc:
        # Stop listener
        return False

listener = Listener(on_press=on_press)
listener.start()

while(True):# This will repeat the indented code below forever   
    time.sleep(0.1)
    
    print('end\n')

