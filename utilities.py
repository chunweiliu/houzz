"""
Some functions we used in programs
"""


import time

import numpy as np
import matplotlib as mpl
mpl.use('Agg')  # while first call
from matplotlib import pyplot as plt
import collections

import houzz

def count():
    """
    Count the chance for Null hypothesis
    """
    counts = collections.defaultdict(float)
    _, test_labels = houzz.partition(n_test=3461, n_train=30000)
    for label in test_labels.values():
        counts[label] += 1.0

    print "chance: {}".format(counts.values()[0]/sum(counts.values()))

def fullfile(root, base):
    """
    Concatinate path in Unix-style
    @param root (str): Unix-style path
    @param base (str): Based name of a unix-style path
    """
    return standardize(root) + '/' + base


def standardize(path):
    """
    Adds trailing '/' if absent.
    @param path (str): Unix-style path
    @return std (str): Unix-style path with trailing '/'
    """
    if not path:
        return None
    return path if path[-1] == '/' else path + '/'


def format_print(text):
    """
    Prints a message along with the current time.
    @param text (str): The meesage
    """
    pattern = ' >> '
    print time.asctime(time.localtime()) + pattern + text


def plot(norm_conf, filename):
    """
    Plot the normalized confusion matrix to the filename
    @param norm_conf (numpy.array): a n by n matrix
    @param filename (str): path to the output png file
    """
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

    fig.colorbar(res)
    labels = ('trad.', 'cont.', 'ecle.', 'modern',
              'med.', 'tropical', 'asian')
    plt.xticks(range(width), labels)
    plt.yticks(range(height), labels)
    plt.savefig(filename, format='png')


def fullfile(root, base):
    """
    Concatinate path in Unix-style
    @param root (str): Unix-style path
    @param base (str): Based name of a Unix-style path
    """
    return standardize(root) + base


def pack_name(img_feature_type, txt_feature_type, pca_k, n_test, n_train,
              output_dir):
    """
    Pack the model's name and meta from the input arguments
    """
    # Specify the model path
    model_name = 'img_{0}_txt_{1}_pca_{2}_test_{3}_train_{4}.model'.format(
        img_feature_type, txt_feature_type, pca_k, n_test, n_train)
    model_path = fullfile(output_dir, model_name)
    # Specify the model meta path
    model_meta = model_path[:-len('.model')] + '.pkl'
    return model_path, model_meta
