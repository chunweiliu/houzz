"""
Subroutines for training the SVM.
"""
import numpy
import collections
import matplotlib
matplotlib.use('Agg')
from matplotlib import mlab

from liblinearutil import *

import houzz
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
              txt_feature_dir=houzz.DATASET_ROOT + 'text_features',
              output_dir=houzz.TRAINED_PATH,
              load_img=True, load_txt=True, pca_k=0):
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
    x, y, img_sf, txt_sf = load_dataset(training_labels, img_feature_dir,
                                        txt_feature_dir, load_img, load_txt)

    # PCA
    if pca_k:
        pca = mlab.PCA(x)
        pca.K = pca_k
        x = pca.Y[:, :pca.K]  # data x projected on the priciple plane
    else:
        pca = None

    x = x.tolist()
    y = y.tolist()

    c = grid_search(y, x)

    format_print("Cross validation complete.")
    format_print("C = {0}\n".format(c))

    # Using the values of 'C' we got from grid-search
    # re-train on the data set.
    # -b 1 -> use probability estimates
    # -q   -> suppress output

    weights = compute_weights(training_labels)
    # weights = ''  # if using blance_partition
    options = "-c {0} -q {1}".format(c, weights)

    format_print(options)
    format_print("Training SVM with cross-validation parameters ...")
    model = train(y, x, options)
    format_print("Training complete.")

    # Save model for the future
    model_name = name + '.model' if not name.endswith('.model') else name
    model_path = output_dir + '/' + model_name
    format_print("Saving model to " + model_path + " ...")
    save_model(model_path, model)
    format_print("Done.")

    return model, img_sf, txt_sf, pca


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
        mat *= 100.0
        for e, a in zip(expected, actual):
            mat[e, a] = round(mat[e, a], 2)
    return mat


def compute_weights(training_labels):
    """Compute weights for SVM based on the number of training data

    @param training_labels (dict)

    @returns weight (string) a counter for the frequency of labels
    """
    counter = collections.defaultdict(int)
    for filename, label in training_labels.iteritems():
        counter[label] += 1

    for label, num_items in counter.iteritems():
        counter[label] = 1 / float(num_items) ** 0.5

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
