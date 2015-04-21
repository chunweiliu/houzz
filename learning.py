"""
Subroutines for training the SVM.
"""
import houzz
import numpy
from liblinearutil import *
import collections

from features import *
from format_print import format_print


def grid_search(y, x, k=5):
    """
    Perform model selection with k-fold cross-validation.

    @param y: list of training labels
    @param x: list of training features
    @param options: string of LIBSVM options
    @param k: number of sets to split the training data into
    @return best c
    @return best gamma
    """

    # format_print("Beginning cross-validation:")
    # x_split, y_split = split_data(x, y, k)
    # format_print("Finished spliting data.")

    """
    Grid search for best c, gamma
    For each c, gamma pair in the grid, try each possible held-out split.
    Take the (c, gamma) that yields the lowest aggregate error rate.
    """

    # Recommended grids from LIBSVM guide
    c_grid = [2**i for i in xrange(-5, 15, 2)]

    best_accuracy = 0.0
    for c in c_grid:

        options = "-c {0} -v {1} -q".format(c, k)
        accuracy = train(y, x, options)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            c_best = c

    format_print(
        """Cross-validation finished!
           c = {0}
           best accuracy = {1}""".format(c, best_accuracy)
    )
    return c_best


def flatten(list_of_lists):
    """
    Subroutine to flatten a list of lists.
    """
    flat = []
    for l in list_of_lists:
        for x in l:
            flat.append(x)
    return flat


def train_svm(name, training_labels, img_feature_dir,
              text_feature_dir=houzz.DATASET_ROOT + 'text_features',
              output_dir=houzz.TRAINED_PATH):
    """
    Train an SVM for each attribute.
    Use 5-fold cross-validation, RBF Kernel.

    @param name (str): name of the classifier
    @param training_labels (dict: str -> int): (filename, label) pairs
    @param img_feature_dir (str): location of the image features
    @param text_feature_dir (str): location of the text features
    @param output_dir (str): where trained SVMs will be saved

    @return LIBSVM model (also writes model file to output_dir)
    """
    # LIBSVM expects features and labels in separate lists
    x, y = load_dataset(training_labels, img_feature_dir)
    # x,y = debug(x,y)  # debug

    c = grid_search(y, x)

    format_print("Cross validation complete.")
    format_print("C = {0}\n".format(c))

    # Using the values of 'C' and 'gamma' we got from cross-validation,
    # re-train on the data set.
    # -b 1 -> use probability estimates
    # -q   -> suppress output
    # RBF kernel by default
    weights = compute_weights(training_labels)
    options = "-c {0} -q {1}".format(c, weights)

    format_print(options)
    format_print("Training SVM with cross-validation parameters ...")
    model = train(y, x, options)
    format_print("Training complete.")

    # Save model for the future
    model_name = name + '.model' if not name.endswith('.model') else name
    format_print("Saving model to " + houzz.TRAINED_PATH + model_name + " ...")
    save_model(houzz.TRAINED_PATH + model_name, model)
    format_print("Done.")

    return model


def load_dataset(names_to_labels, img_feature_dir):
    """
    @param names_to_labels (dict: str -> int)
    @param img_feature_dir (str)
    @return list of features
    @return list of labels (same order as list of features)
    """
    x, y = [], []
    for stem in names_to_labels.keys():
        x.append(feature(stem, img_feature_dir))
        y.append(names_to_labels[stem])
    return x, y


def confusion(expected, actual, percentages=True):
    """
    Generate the confusion matrix for test results.

    @param expected: list of ground-truth labels
    @param actual: list of predicted labels
    @param percentages: True  -> table of percentages
                        False -> table of counts
    @return numpy array where entry (i, j) is
            prediction of label j with ground truth i
    """
    n = len(houzz.LABELS)
    mat = numpy.zeros((n, n))

    # Record the counts for each prediction
    totals = numpy.zeros(n)
    for e, a in zip(expected, actual):
        mat[e, a] += 1
        totals[e] += 1

    # Make entries into percentages
    if percentages:
        for e in xrange(n):
            mat[e] /= (totals[e] if totals[e] else 1)
            # Distribution of predictions when truth was e
    return mat


def compute_weights(training_labels):
    """Compute weights for SVM based on the number of training data

    @param training_labels (dict)

    @returns weight (string) a counter for the frequency of labels
    """
    counter = collections.defaultdict(int)
    for filename, label in training_labels.iteritems():
        counter[label] += 1

    for key, value in counter.iteritems():
        counter[key] = 1 / value ** 0.5

    weights = ''
    for key, value in counter.iteritems():
        weights += '-w{0} {1} '.format(key, value)
    return weights

def debug(x, y):
    """DEBUG"""
    x_return = []
    y_return = []
    for i, j in zip(x, y):
        if j == 0:
            continue
        else:
            x_return.append(i)
            y_return.append(j)
    return x_return, y_return