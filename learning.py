"""
Subroutines for training the SVM.
"""

from random import randint
import sys
from svmutil import *


def cross_validation(y, x, options):
    """
    Performs cross-validation on a 70-30 split of the training data.

    @param 
        y: list of training labels
        x: list of training features
        options: string of LIBSVM options
    @returns 
        c, gamma parameters
    """

    # Randomly split data
    x_train, y_train = [], []
    x_hold,  y_hold  = [], []
    for i in xrange(0, len(y)):
        roll = randint(1, 10)
        if roll <= 7:
            x_train.append(x[i])
            y_train.append(y[i])
        else:
            x_hold.append(x[i])
            y_hold.append(y[i])

    # Recommended grids from LIBSVM guide
    c_grid     = [2**x for x in xrange(-5, 15, 2)]
    gamma_grid = [2**x for x in xrange(-15, 3, 2)] 
    
    best_accuracy = 0.0
    for c in c_grid:
        for gamma in gamma_grid:
            # LIBSVM options:
            # -c <cost  parameter>
            # -g <gamma parameter>            
            options = "-c {0} -g {1}".format(c, gamma)
            model = svm_train(y_train, x_train, options)
            predict, stats, prob = svm_predict(y_hold, x_hold, model)
            accuracy = stats[0]

            if accuracy > best_accuracy:
                best_accuracy = accuracy 
                c_best = c
                gamma_best = gamma
   
    return c_best, gamma_best

