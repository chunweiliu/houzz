import time


def format_print(text):
    """ Provide useful information for print
    Time text
    """
    pattern = ' >> '
    print time.asctime(time.localtime()) + pattern + text
