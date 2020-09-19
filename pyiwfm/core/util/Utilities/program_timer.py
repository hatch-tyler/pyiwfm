'''
program_timer.py
Author: Tyler Hatch PhD, PE

This is the program_timer module of the python version of IWFM
'''
# std library imports
import time

# set module variables
timer_stopped = False
timer_started = False

def start_timer():
    ''' records start time 
    Parameters
    ----------
    None
    
    Returns
    -------
    float
        time in seconds since the Epoch 
    '''
    global timer_started
    timer_started = True
    
    return time.time()

def stop_timer():
    ''' records stop time
    Parameters
    ----------
    None
    
    Returns
    -------
    float
        time in seconds since the Epoch 
    '''
    global timer_stopped
    timer_stopped = True

    return time.time()

def get_run_time(start_time, stop_time):
    ''' calculates the program run time from
    the start time and stop time.

    Parameters
    ----------
    start_time : float
        starting clock time in seconds since epoch. see time.time

    stop_time : float
        stopping clock time in seconds since epoch. see time.time

    Returns
    -------
    tuple (int, int, float)
        hours, minutes, seconds

    Usage
    -----
    >>> start = start_timer()
    >>> time.sleep(10)
    >>> stop = stop_timer()
    >>> get_run_time(start, stop)
    ... (0, 0, 10.000308990478516)
    '''

    if not timer_started:
        return 0, 0, 0

    # calculate duration in seconds
    duration = stop_time - start_time

    # convert duration to hours, minuts, seconds
    minutes, seconds = divmod(duration, 60)
    hours, minutes = divmod(minutes, 60)

    return int(hours), int(minutes), seconds

def int_to_text(number):
    ''' converts a number to a string '''
    return str(number)
    

if __name__ == "__main__":
    
    start_time_values = start_timer()

    time.sleep(10)
    
    end_time_values = stop_timer()

    hour, minute, second = get_run_time(start_time_values, end_time_values)

    print(hour, minute, second)