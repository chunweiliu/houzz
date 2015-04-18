"""
Subroutines for training the SVM.
"""

import random
import sys
from svmutil import *

from format_print import format_print


def cross_validation(y, x, options, k=5):
    """
    Performs k-fold cross-validation.

    @param 
        y: list of training labels
        x: list of training features
        options: string of LIBSVM options
        k: number of sets to split the training data into
    @returns 
        c, gamma parameters
    """
    
    format_print("Beginning cross-validation:")
    x_split, y_split = split_data(x, y, k) 
    format_print("Finished spliting data.")
    
    """
    Grid search for best c, gamma
    For each c, gamma pair in the grid, try each possible held-out split.
    Take the (c, gamma) that yields the lowest aggregate error rate.
    """

    # Recommended grids from LIBSVM guide
    c_grid     = [2**x for x in xrange(-5, 15, 2)]
    gamma_grid = [2**x for x in xrange(-15, 3, 2)] 
    
    best_accuracy = 0.0
    for c, gamma in zip(c_grid, gamma_grid):
        avg_accuracy = 0.0
        for i in xrange(k):
        
            format_print("Cross-validating with held-out set {0}, "
                         "c = {1}, gamma = {2}.".format(k, c, gamma))

            x_hold = x_split[i]
            y_hold = y_split[i]
            
            x_train = flatten([split for split in x_split if split is not x_hold])
            y_train = flatten([split for split in y_split if split is not y_hold])

            # LIBSVM options:
            # -c <cost  parameter>
            # -g <gamma parameter>            
            options = "-c {0} -g {1}".format(c, gamma)
            model = svm_train(y_train, x_train, options)
            predicted_labels, _, _ = svm_predict(y_hold, x_hold, model)
            
            # compute number of incorrect labels
            num_wrong = 0
            for i, label in enumerate(y_hold):
                if label != predicted_labels[i]:
                    num_wrong += 1
        
            avg_accuracy += num_wrong  # keep a running total

        avg_accuracy /= k  # avg_accuracy is type float

        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy 
            c_best = c
            gamma_best = gamma

        
    format_print(
    """Cross-validation finished!
        c = {0}
        gamma = {1}
        best accuracy = {2}""".format(c, gamma, best_accuracy)
    )
    return c_best, gamma_best


def split_data(x, y, k):
    # k splits
    x_split = [[] for i in range(k)]  
    y_split = [[] for i in range(k)]
    # Why not [[]]*k ?
    # It reuses the same reference for [] each time!

    # Randomly permute indices and count off 0, 1, ..., k-1
    # assert len(y) == len(x)
    indices = range(len(y))
    random.shuffle(indices)
    for i in xrange(len(y)):
        fold_number = i % k  # cycles through 0, 1, ..., k-1
        x_split[fold_number].append(x[indices[i]])
        y_split[fold_number].append(y[indices[i]])

    return x_split, y_split


def flatten(list_of_lists):
    """
    Subroutine to flatten a list of lists.
    """
    flat = []
    for l in list_of_lists:
        for x in l:
            flat.append(x)
    return flat
