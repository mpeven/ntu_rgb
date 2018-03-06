from __future__ import print_function
import sys
import time
import datetime as dt
import threading

class ProgressMeter:
    def __init__(self):
        self.progress_meter = self.spinning_cursor()


    def start(self, text=None, timer=False):
        '''
        Starts the progress meter in another thread
        '''
        self.thread = threading.Thread(target=self.print_progress, args=(text, timer))
        self.thread.start()


    def stop(self):
        '''
        Stop the progress meter
        '''
        self.keep_running = False
        self.thread.join()


    def print_progress(self, text, timer):
        '''
        Prints out the currently set progress meter to the command line
        '''
        self.keep_running = True
        if text:
            print(text, end="  ", flush=True)
        if timer:
            start_time = dt.datetime.now()
        try:
            while self.keep_running:
                sys.stdout.write(next(self.progress_meter))
                sys.stdout.flush()
                time.sleep(0.1)
                sys.stdout.write(next(self.progress_meter))
            if timer:
                print("Done {}".format(dt.datetime.now() - start_time))
        except KeyboardInterrupt:
            return



    ##############################
    #      Progress meters       #
    ##############################

    def spinning_cursor(self):
        while True:
            for cursor in '|/-\\':
                yield cursor
                yield '\b'


    def dots(self):
        dot_groups = ['', '.', '\b', '..', '\b\b', '...', '\b\b\b', '....', '\b\b\b\b']
        while True:
            for dot_group in dot_groups:
                yield dot_group

    ##############################




# Test
def long_running_function():
    a = 0
    for x in range(100000000):
        if x%2 == 0:
            a += x
        else:
            a -= x

if __name__ == '__main__':
    spinning_cursor = ProgressMeter()
    spinning_cursor.start("Processing some data", True)
    long_running_function()
    spinning_cursor.stop()
