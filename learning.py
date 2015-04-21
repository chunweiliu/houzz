"""
Subroutines for training the SVM.
"""
import houzz
import random
import numpy
from svmutil import *
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
    c_grid     = [2**i for i in xrange(-5, 15, 2)]
    gamma_grid = [2**i for i in xrange(-15, 3, 2)]

    best_accuracy = 0.0
    for c, gamma in zip(c_grid, gamma_grid):
        # avg_accuracy = 0.0
        # for i in xrange(k):

        #     format_print("Cross-validating with held-out set {0}, "
        #                  "c = {1}, gamma = {2}.".format(i, c, gamma))

        #     x_hold = x_split[i]
        #     y_hold = y_split[i]

        #     x_train = flatten([split for split in x_split if split is not x_hold])
        #     y_train = flatten([split for split in y_split if split is not y_hold])

        #     # LIBSVM options:
        #     # -c <cost  parameter>
        #     # -g <gamma parameter>
        #     # -q suppress output
        #     options = "-c {0} -g {1} -q".format(c, gamma)
        #     model = svm_train(y_train, x_train, options)
        #     predicted_labels, _, _ = svm_predict(y_hold, x_hold, model)

        #     # compute number of incorrect labels
        #     num_wrong = 0
        #     for i, label in enumerate(y_hold):
        #         if label != predicted_labels[i]:
        #             num_wrong += 1

        #     avg_accuracy += num_wrong  # keep a running total

        # avg_accuracy /= k  # avg_accuracy is type float

        options = "-c {0} -g {1} -v {2} -q".format(c, gamma, k)
        accuracy = svm_train(y, x, options)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
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

    c, gamma = grid_search(y, x)

    format_print("Cross validation complete.")
    format_print("C = {0}, gamma = {1}\n".format(c, gamma))

    # Using the values of 'C' and 'gamma' we got from cross-validation,
    # re-train on the data set.
    # -b 1 -> use probability estimates
    # -q   -> suppress output
    # RBF kernel by default

    weights = compute_weights(training_labels)
    options = "-c {0} -g {1} -b 1 -q {2}".format(c, gamma, weights)

    # options = "-c {0} -g {1} -b 1 -q".format(c, gamma)
    format_print(options)
    format_print("Training SVM with cross-validation parameters ...")
    model = svm_train(y, x, options)
    format_print("Training complete.")

    # Save model for the future
    model_name = name + '.model' if not name.endswith('.model') else name
    format_print("Saving model to " + houzz.TRAINED_PATH + model_name + " ...")
    svm_save_model(houzz.TRAINED_PATH + model_name, model)
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


# def test(model, test_labels, img_feature_dir):
#     """
#     @param model: LIBSVM model
#     @param test_labels (dict: str -> int)
#     @param img_feature_dir (str)
#     @return list of predicted labels
#     @return accuracy
#     """
#     x, y = load_dataset(test_labels, img_feature_dir)
#     x,y = debug(x,y)  # debug
#     print y.count(0)
#     print y.count(1)

#     labels, stats, _ = svm_predict(y, x, model, '-q')
#     return labels, stats[0]  # accuracy


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