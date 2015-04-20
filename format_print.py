import time


def format_print(text):
    """ Prints a message along with the current time."""
    pattern = ' >> '
    print time.asctime(time.localtime()) + pattern + text
