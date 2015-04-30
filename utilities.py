"""
Some functions we used in programs
"""


import time

import numpy as np
# import matplotlib as mpl
# mpl.use('Agg')  # while first call
from matplotlib import pyplot as plt


def standardize(path):
    """
    Adds trailing '/' if absent.
    @param path (str): Unix-style path
    @return std (str): Unix-style path with trailing '/'
    """
    return path if path[-1] == '/' else path + '/'


def format_print(text):
    """
    Prints a message along with the current time.
    @param text (str): The meesage
    """
    pattern = ' >> '
    print time.asctime(time.localtime()) + pattern + text


def plot(norm_conf, filename):

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,
                    interpolation='nearest')

    width = len(norm_conf)
    height = len(norm_conf[0])

    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(norm_conf[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res)
    labels = ('trad.', 'cont.', 'ecle.', 'modern',
              'med.', 'tropical', 'asian')
    plt.xticks(range(width), labels)
    plt.yticks(range(height), labels)
    plt.savefig(filename, format='png')
