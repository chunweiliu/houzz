"""
Subroutines for training the SVM.
"""
import numpy as np
import collections
import cPickle
import matplotlib
matplotlib.use('Agg')
from matplotlib import mlab

import liblinearutil

import houzz
from utilities import format_print
from utilities import fullfile
from utilities import pack_name
from utilities import standardize

import os


def train(img_feature_type, txt_feature_type, feature_dir, pca_k,
          n_test, n_train, train_labels, svm_dir):
    """
    Given the type of feautres, train a svm
    @param (str) img_feature_type
    @param (str) txt_feature_type
    @param (str) feature_dir
    @param (str) pca_k, the number of principle component returns in PCA
    @param (str) n_train, number of training data
    @param (str) n_test, number of test data
    @param (str) svm_dir, the place to store the pkl and model files
    """
    model_path, model_meta = pack_name(img_feature_type, txt_feature_type,
                                       pca_k, n_test, n_train, svm_dir)

    # Train a model if there are no such model and meta existed
    if not os.path.exists(model_path) and not os.path.exists(model_meta):

        # Get the image and text feature paths
        img_feature_dir = None if not img_feature_type else \
            standardize(fullfile(feature_dir, 'img/' + img_feature_type))

        txt_feature_dir = None if not txt_feature_type else \
            standardize(fullfile(feature_dir, 'txt/' + txt_feature_type))

        # Train the SVM (need to refactorize)
        model, pca, img_sf, txt_sf = train_svm(train_labels, img_feature_dir,
                                               txt_feature_dir)

        # Save the trained model and its meta
        liblinearutil.save_model(model_path, model)

        # Create a record for the training results
        meta = Meta()
        meta.img_sf = img_sf
        meta.txt_sf = txt_sf
        meta.pca = pca
        meta.train_labels = train_labels
        meta.img_feature_dir = img_feature_dir
        meta.txt_feature_dir = txt_feature_dir

        with open(model_meta, 'w') as f:
            cPickle.dump(meta, f)
    else:
        model = liblinearutil.load_model(model_path)
        with open(model_meta, 'r') as f:
            meta = cPickle.load(f)

    return model, meta


def test(model, meta, test_labels):
    """
    Test pretrained SVM
    @returns (array) mat, a normalized confusion matrix
    @returns (float) acc, an accuray of the test labels
    """
    # Unpack meta record
    img_sf = meta.img_sf
    txt_sf = meta.txt_sf
    pca = meta.pca
    img_feature_dir = meta.img_feature_dir
    txt_feature_dir = meta.txt_feature_dir

    # Testing
    format_print("Testing ...")
    img_features, txt_features = load_features(
        test_labels, img_feature_dir, txt_feature_dir)

    # Scale the features
    img_features = scale(img_features, img_sf)
    txt_features = scale(txt_features, txt_sf)

    # PCA
    if pca and img_features is not None and img_features.any():
        img_features = pca.project(img_features)
        img_features = img_features[:, :pca.k]

    # Concatenate scaled features as np array
    x, y = [], []
    for img, txt, stem in zip(img_features, txt_features, test_labels):
        x.append(img.tolist() + txt.tolist())
        y.append(test_labels[stem])

    predicted_labels, stats, p_val = liblinearutil.predict(y, x, model, '-q')

    acc = stats[0]
    mat = confusion(y, predicted_labels)
    return mat, acc


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
        accuracy = liblinearutil.train(y, x, options)
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


def find_scale_factor(data):
    """Find the scale factor of the training dataset

    @param data (list or ndarray of ndarrays) features from training data
    @return sf (double) the scale factor that scales the training data set
    to [-1, 1]
    """
    if data is None or not data.any():
        return None

    normalizer = max(abs(data.max()), abs(data.min()))
    return float(normalizer) if normalizer > 10e-9 else 1  # don't scale


def scale(data, normalizer):
    """
    Scale a nested ndarray of features.

    @param data (ndarray)
    @return (ndarray)
    """
    if data is None or not normalizer:
        return data

    return data / normalizer


def train_svm(training_labels, img_feature_dir=None, txt_feature_dir=None,
              pca_k=300):
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
    img_features, txt_features = load_features(
        training_labels, img_feature_dir, txt_feature_dir)

    # Find scale factors, scale the features
    img_sf = find_scale_factor(img_features)
    txt_sf = find_scale_factor(txt_features)

    img_features = scale(img_features, img_sf)
    txt_features = scale(txt_features, txt_sf)

    # Apply PCA
    if img_features is not None and img_features.any():
        try:
            pca = mlab.PCA(img_features)
            pca.k = pca_k
            img_features = pca.Y[:, :pca_k]
        except ValueError:
            print "Something wrong ..."
            pca = None
    else:
        pca = None

    # Concatenate scaled features as np array
    x, y = [], []
    for img, txt, stem in zip(img_features, txt_features, training_labels):
        x.append(img.tolist() + txt.tolist())
        y.append(training_labels[stem])

    c = grid_search(y, x)

    format_print("Cross validation complete.")

    # Using the values of 'C' we got from grid-search
    # re-train on the data set.
    # -b 1 -> use probability estimates
    # -q   -> suppress output

    weights = compute_weights(training_labels)
    # weights = ''  # if using blance_partition
    options = "-c {0} -q {1}".format(c, weights)

    format_print(options)
    format_print("Training SVM with cross-validation parameters ...")
    model = liblinearutil.train(y, x, options)
    format_print("Training complete.")

    return model, pca, img_sf, txt_sf


def confusion(expected, actual, percentages=True):
    """
    Generate the confusion matrix for test results.

    @param expected: list of ground-truth labels
    @param actual: list of predicted labels
    @param percentages: True  -> table of percentages
                        False -> table of counts
    @return np array where entry (i, j) is
            prediction of label j with ground truth i
    """
    n = len(houzz.LABELS)
    mat = np.zeros((n, n))

    # Record the counts for each prediction
    totals = np.zeros(n)
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


def load_features(names_to_labels, img_dir=None, txt_dir=None):
    """
    Combine individually scaled image and text features for the dataset.

    Preconditions:
        1) Image features precomputed and stored in img_dir
        2) Text features precomputed and stored in txt_dir

    @param names_to_labels (dict: str -> int)
    @param img_dir (str)
    @param txt_dir (str)

    @return img_features (ndarray)
    @return txt_features (ndarray)
    """

    img_dir = standardize(img_dir)
    txt_dir = standardize(txt_dir)

    img_features = []
    txt_features = []

    # Load all image and text features to compute scaling
    names = names_to_labels.keys()
    for stem in names:
        if img_dir:
            img = np.load(img_dir + stem + '.npy').flatten()
        else:
            img = np.array([])

        if txt_dir:
            txt = np.load(txt_dir + stem + '.npy').flatten()
        else:
            txt = np.array([])

        img_features.append(img)
        txt_features.append(txt)

    # Turn lists to ndarrays for scaling
    img_features = np.array(img_features)
    txt_features = np.array(txt_features)

    return img_features, txt_features


class Meta:
    """Record for the associated data for a trained model."""
    pass


def load_model(model_path, model_meta):
    model = liblinearutil.load_model(model_file)
    with open(model_meta, 'r') as f:
        meta = cPickle.load(f)
    return model, meta
