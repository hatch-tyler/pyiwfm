""" 
messagelogger.py
Author: Tyler Hatch, PhD, PE

This is the message_logger module of the python version of IWFM

Notes
-----
All Caps used for parameters that should not be changed

functions and subroutines from the IWFM fortran code are replicated
here but some of them may not be used in the python construct and will
raise a NotImplementedError. 

e.g. GetLogFileUnit in IWFM doesn't have a python equivalent since python 
does not reference files using units.
"""
# std library imports
import os

# local imports
from pyiwfm.core.util.Utilities import program_timer

# global settings for echoing program progress
# these should not be changed during runtime
YES_ECHO_PROGRESS = 1
NO_ECHO_PROGRESS = 0

# this can be changed at runtime
flag_echo_progress = NO_ECHO_PROGRESS

# global settings for message destination
# these should not be changed during runtime
SCREEN_FILE = 1
SCREEN = 2
FILE = 3

# settings for message severity levels
# these should not be changed during runtime
MESSAGE = 0
INFO = 1
WARN = 2
FATAL = 3

# module variables for last saved message
last_message = ""
last_message_type = INFO
last_message_procedure = ""

# data definitions
THIS_PROCEDURE = "MessageLogger"
LINE_FEED = "\n"
DEFAULT_LOG_FILE_NAME = "Message.log"
message_array = []
warnings_generated = False

# this is not really needed for python
console_exists = True

default_message_destination = SCREEN_FILE


def make_log_file(log_file_name=DEFAULT_LOG_FILE_NAME):
    """function to open a log file in write mode.
    if log_file does not exist it will be created.
    if log_file exists it will be overwritten."""
    global log_file
    log_file = open(log_file_name, "w")


def kill_log_file():
    """function to close log_file"""
    try:
        log_file.close()
    except NameError:
        pass


def set_flag_to_echo_progress(flag):
    """sets the global variable flag_echo_progress

    Parameters
    ----------
    flag : int
        must be 0 (NO_ECHO_PROGRESS) or 1 (YES_ECHO_PROGRESS)

    Returns
    -------
    None
    """
    if flag != YES_ECHO_PROGRESS and flag != NO_ECHO_PROGRESS:
        set_last_message(
            "flag to echo progress is not recognized", FATAL, THIS_PROCEDURE
        )
    else:
        global flag_echo_progress
        flag_echo_progress = flag


def set_default_message_destination(destination):
    """sets the default message destination to print to
    the console, log file or both.

    Parameters
    ----------
    destination : int
        must be 1 (SCREEN_FILE), 2 (SCREEN), or 3 (FILE)

    Returns
    -------
    None
        sets global variable default_message_destination
    """
    test = (
        (destination == SCREEN) or (destination == FILE) or (destination == SCREEN_FILE)
    )
    if not test:
        set_last_message(
            "message destination is not recognized!", FATAL, THIS_PROCEDURE
        )

    global default_message_destination
    default_message_destination = destination


def set_log_file_name(file_name):
    """sets the name of the log file by calling the make_log_file function

    Parameters
    ----------
    file_name : str
        name of the log file

    Returns
    -------
    None
        creates and opens a new log file with the global variable name log_file.
    """
    try:
        if log_file:
            set_last_message(
                "Error in opening new log file! A log file is already created!",
                FATAL,
                THIS_PROCEDURE,
            )
    except NameError:
        make_log_file(file_name)


def set_last_message(message, error_level, program_name):
    """this is the combination of the IWFM SetLastMessage_Array and
    SetLastMessage_Single. This takes advantage of Python's flexible
    data structures.

    Parameters
    ----------
    message : str or list of str
        message set as last message

    error_level : int
        message severity levels MESSAGE, INFO, WARN, FATAL

    program_name : str
        name of program or procedure where message originates

    Returns
    -------
    None
        sets global variables for last_message, last_message_procedure, and last_message_type
    """
    global last_message
    global last_message_procedure
    global last_message_type

    if len(message) == 1 or isinstance(message, str):

        last_message = "*   " + message

    else:
        message = ["*   " + val for val in message]
        last_message = LINE_FEED.join(message)

    last_message_procedure = program_name

    last_message_type = error_level


def get_last_message():
    """returns the formatted last message"""
    if last_message_type == INFO:
        return "* INFO:" + LINE_FEED + last_message
    elif last_message_type == WARN:
        return "* WARN:" + LINE_FEED + last_message
    elif last_message_type == FATAL:
        return "* FATAL:" + LINE_FEED + last_message


def get_log_file_unit(file_name):
    raise NotImplementedError(
        "This construct is not used in the" + "python implementation of IWFM"
    )


def get_file_unit_number(file_name):
    raise NotImplementedError(
        "This construct is not used in the" + "python implementation of IWFM"
    )


def is_log_file_defined():
    raise NotImplementedError(
        "This construct is not used in the" + "python implementation of IWFM"
    )


def get_a_unit_number():
    raise NotImplementedError(
        "This construct is not used in the" + "python implementation of IWFM"
    )


def primitive_error_handler(message):
    raise NotImplementedError(
        "This construct is not used in the" + "python implementation of IWFM"
    )


def log_all_message_types(message_array, error_level, prog_name, destination):
    """logs all messages according to level of severity to the console and/or log file
    depending on settings

    Parameters
    ----------
    message_array: list
        list of messages to be written

    error_level : int
        message severity level i.e. MESSAGE, INFO, WARN, OR FATAL

    prog_name : str
        name of the program/procedure where messages originated

    destination : int
        message output destination i.e. SCREEN_FILE, SCREEN, OR FILE

    Returns
    -------
    None
        writes information to the console or log file
    """

    def print_message_array(messages, dest):
        """function to write message to its destination"""
        if dest == FILE:
            for msg in messages:
                log_file.write(msg + "\n")
        if dest == SCREEN:
            for msg in messages:
                print(msg)

    global warnings_generated

    if destination == SCREEN_FILE:
        will_print_to_file = True
        will_print_to_screen = True
    elif destination == FILE:
        will_print_to_file = True
        will_print_to_screen = False
    elif destination == SCREEN:
        will_print_to_file = False
        will_print_to_screen = True
    else:
        will_print_to_file = False
        will_print_to_screen = False

    several_messages = message_array

    if destination != SCREEN:
        try:
            # attempt to check if log file is closed
            # if log_file variable doesn't exist will raise a NameError
            if log_file.closed:
                pass  # may want to open log_file if variable exists, but is closed

        except NameError:
            make_log_file()

    if error_level != 0:
        several_messages = ["*   " + msg for msg in message_array]

    if error_level == MESSAGE:
        if will_print_to_file:
            print_message_array(several_messages, FILE)
        if will_print_to_screen:
            print_message_array(several_messages, SCREEN)

    elif error_level == INFO:
        warnings_generated = True

        if will_print_to_file:
            log_file.write("* INFO : \n")
            print_message_array(several_messages, FILE)
            log_file.write("*   ({})\n".format(prog_name))

        print("* INFO : ")
        print_message_array(several_messages, SCREEN)
        print("*   ({})".format(prog_name))

    elif error_level == WARN:
        warnings_generated = True

        if will_print_to_file:
            log_file.write("* WARN : \n")
            print_message_array(several_messages, FILE)
            log_file.write("*   ({})\n".format(prog_name))

        print("* WARN : ")
        print_message_array(several_messages, SCREEN)
        print("*   ({})".format(prog_name))

    elif error_level == FATAL:

        if will_print_to_file:
            log_file.write(" ")
            log_file.write(
                "*******************************************************************************"
            )
            log_file.write("* FATAL: ")
            print_message_array(several_messages, FILE)
            log_file.write("*   ({})".format(prog_name))
            log_file.write(
                "*******************************************************************************"
            )

        print(" ")
        print(
            "*******************************************************************************"
        )
        print("* FATAL: ")
        print_message_array(several_messages, SCREEN)
        print("*   ({})".format(prog_name))
        print(
            "*******************************************************************************"
        )

    else:
        if will_print_to_file:
            log_file.write(" ")
            log_file.write(
                "*******************************************************************************"
            )
            log_file.write("*")
            log_file.write("* FATAL:")
            log_file.write(
                "*   Incorrect error level returned from procedure {}".format(prog_name)
            )
            log_file.write("*   ({})".format(prog_name))
            log_file.write("*")
            log_file.write(
                "*******************************************************************************"
            )

        print(" ")
        print(
            "*******************************************************************************"
        )
        print("*")
        print("* FATAL:")
        print("*   Incorrect error level returned from procedure {}".format(prog_name))
        print('*   ("{}")'.format(prog_name))
        print("*")
        print(
            "*******************************************************************************"
        )


def log_message(
    message, error_level, prog_name, destination=default_message_destination
):
    """This is the combination of the IWFM LogSingleMessage and LogMessageArray.
    This takes advantage of python's flexible data structures

    Parameters
    ----------
    message : str or list of str
        message(s) to be logged

    error_level : int
        message severity level i.e. MESSAGE, INFO, WARN, OR FATAL

    prog_name : str
        name of the program/procedure where messages originated

    destination : int
        message output destination i.e. SCREEN_FILE, SCREEN, OR FILE

    Returns
    -------
    None
        calls log_all_message_types
    """
    if isinstance(message, str):
        local_message = [message]
        log_all_message_types(local_message, error_level, prog_name, destination)

    if isinstance(message, list):
        log_all_message_types(message, error_level, prog_name, destination)


def log_last_message(destination=default_message_destination):
    """This function logs the last message. This is the one used in the IWFM main program.
    Parameters
    ----------
    destination : int, default=default_message_destination
        message output destination i.e. SCREEN_FILE, SCREEN, OR FILE

    Returns
    -------
    None
        logs last_message
    """
    message_local = "*******************************************************************************"

    if last_message_type == INFO:
        message_local = "{}{}* INFO: ".format(message_local, LINE_FEED)
    elif last_message_type == WARN:
        message_local = "{}{}* WARN: ".format(message_local, LINE_FEED)
    elif last_message_type == FATAL:
        message_local = "{}{}* FATAL: ".format(message_local, LINE_FEED)

    message_local = "{}{}{}".format(message_local, LINE_FEED, last_message)
    message_local = "{}{}*   ({}){}{}".format(
        message_local, LINE_FEED, last_message_procedure, LINE_FEED, "*" * 80
    )

    log_message(message_local, MESSAGE, "", destination)


def check_console_availability():
    raise NotImplementedError(
        "This construct is not used in the" + "python implementation of IWFM"
    )


def print_run_time(
    start_time=None, stop_time=None, destination=default_message_destination
):
    """prints the formatted run time of the simulation

    Parameters
    ----------
    start_time : float or None
        start time of the program

    stop_time : float or None
        stop time of the program

    destination : int, default=default_message_destination
        message output destination i.e. SCREEN_FILE, SCREEN, OR FILE

    Returns
    -------
    None
        writes formatted run time to output destination
    """
    # get stop time if timer_stopped is False
    if not program_timer.timer_stopped:
        stop_time = program_timer.stop_timer()

    # get the run time
    hours, minutes, seconds = program_timer.get_run_time(start_time, stop_time)

    # format results in message_array
    if len(message_array) == 0:
        message_array.append("{}{}".format(LINE_FEED, "*" * 50))
    else:
        message_array[0] = "{}{}".format(LINE_FEED, "*" * 50)

    if len(message_array) <= 1:
        message_array.append("TOTAL RUN TIME: ")
    else:
        message_array[1] = "TOTAL RUN TIME: "

    if hours > 0:
        message_array[1] = "{}{} HOURS {} MINUTES {:6.3f} SECONDS".format(
            message_array[1], hours, minutes, seconds
        )
    elif minutes > 0:
        message_array[1] = "{}{} MINUTES {:6.3f} SECONDS".format(
            message_array[1], minutes, seconds
        )
    else:
        message_array[1] = "{}{:6.3f} SECONDS".format(message_array[1], seconds)

    if warnings_generated:
        if len(message_array) <= 2:
            message_array.append(
                "WARNINGS/INFORMATIONAL MESSAGES ARE GENERATED!{}".format(LINE_FEED)
            )
        else:
            message_array[
                2
            ] = "WARNINGS/INFORMATIONAL MESSAGES ARE GENERATED!{}".format(LINE_FEED)

        try:
            log_file_name = log_file.name
        except:
            pass
        else:
            message_array[2] = "{}FOR DETAILS CHECK FILE '{}'.{}".format(
                message_array[2], log_file_name, LINE_FEED
            )
        message_array[2] = "{}{}".format(message_array[2], "*" * 50)
    else:
        if len(message_array) <= 2:
            message_array.append("*" * 50)
        else:
            message_array[2] = "*" * 50

    # log the message to the destination
    log_message(message_array, MESSAGE, "", destination)


def close_message_file():
    kill_log_file()


def echo_progress(text):
    if flag_echo_progress == YES_ECHO_PROGRESS:
        log_message(text, MESSAGE, "")


if __name__ == "__main__":
    start_time = program_timer.start_timer()
    # make_log_file('test_messages.log')
    # print(last_message)
    set_log_file_name("test_messages.log")
    # print(last_message)
    # print(log_file.closed)
    # print(warnings_generated)
    log_message(
        "Node number does not have any surrounding elements!", WARN, THIS_PROCEDURE
    )
    set_last_message(
        "An application grid that is already defined is being re-defined!",
        FATAL,
        THIS_PROCEDURE,
    )
    # print(warnings_generated)
    log_last_message()
    stop_time = program_timer.stop_timer()
    print_run_time(start_time, stop_time)
    kill_log_file()
    # print(log_file.closed)
