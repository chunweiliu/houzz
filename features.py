"""
Subroutines for computing and saving features.
"""

import os
import string
import cPickle as pickle
import numpy as np

import gensim
from nltk.corpus import stopwords

from utilities import format_print
from utilities import standardize
from utilities import fullfile
import houzz

import caffe
from caffe.proto import caffe_pb2
import lmdb

# GIST
# import leargist
from PIL import Image

# Color histogram
from colorsys import rgb_to_hsv
from scipy.misc import imread

import matplotlib
matplotlib.use('Agg')
from matplotlib import mlab

def text_feature(meta_folder, text_model_file, text_folder, feature_of):
    """Precompute all textual features for the files in the meta_folder."""

    meta_folder = standardize(meta_folder)
    text_folder = standardize(text_folder)

    # Load the langauge model
    format_print('Loading the Google News Model ...')
    model = gensim.models.Word2Vec.load_word2vec_format(
        text_model_file, binary=True)  # the Google News Model
    format_print('... Finish loading')

    for pkl in os.listdir(meta_folder):
        with open(meta_folder + pkl, 'r') as f:
            metadata = pickle.load(f)
            # feature = compute_text_feature(metadata, model)
            feature = feature_of(metadata, model)
            # Discard empty features
            try:
                if feature.any() is not False:
                    npy = pkl[:-len('.pkl')] + '.npy'
                    np.save(text_folder + npy, feature)
            except AttributeError:
                pass


def compute_tags_feature(metadata, model):
    """Compute an unified vector to represent the description and tags.
    Treat each word in the description and tags the same way.

    @param metadata: a metadata dictionary from Houzz.loadmat
    @return normed feature vector
    """

    if not metadata['tag']:
        format_print('No tag feature found')
        return None

    word_list = []
    if metadata['tag']:
        for tag in metadata['tag']:
            word_list += process_text(tag)

    # Combine the word2vec for each individual word
    text_vector = np.zeros(300, dtype=np.float32)
    text_count = 0
    for word in word_list:
        if word in model:
            text_vector += model[word]
            text_count += 1.0

    text_vector = text_vector / text_count if text_count else text_vector
    return text_vector


def compute_description_feature(metadata, model):
    """Compute an unified vector to represent the description and tags.
    Treat each word in the description and tags the same way.

    @param metadata: a metadata dictionary from Houzz.loadmat
    @return normed feature vector
    """

    if not metadata['description']:
        format_print('No description feature found')
        return None

    word_list = []
    if metadata['description']:
        for word in metadata['description'].split():
            word_list += process_text(word)

    # Combine the word2vec for each individual word
    text_vector = np.zeros(300, dtype=np.float32)
    text_count = 0
    for word in word_list:
        if word in model:
            text_vector += model[word]
            text_count += 1.0

    text_vector = text_vector / text_count if text_count else text_vector
    return text_vector


def compute_text_feature(metadata, model):
    """Compute an unified vector to represent the description and tags.
    Treat each word in the description and tags the same way.

    @param metadata: a metadata dictionary from Houzz.loadmat
    @return normed feature vector
    """

    if not metadata['tag'] and not metadata['description']:
        format_print('No textual feature found')
        return None

    word_list = []
    if metadata['tag']:
        for tag in metadata['tag']:
            word_list += process_text(tag)
    if metadata['description']:
        for word in metadata['description'].split():
            word_list += process_text(word)

    # Combine the word2vec for each individual word
    text_vector = np.zeros(300, dtype=np.float32)
    text_count = 0
    for word in word_list:
        if word in model:
            text_vector += model[word]
            text_count += 1.0

    text_vector = text_vector / text_count if text_count else text_vector
    return text_vector


def process_text(text):
    """
    Preprocess the text being handed to word2vec.

    Each tag in the dataset is a phrase joined by hyphens. For example,

        guest-room-retreat

    This will not appear in the word2vec lexicon. Therefore,
    split each tag into its constituent words and remove stop words.
    Without a special corpus of interior design terms to train the
    language model on, this seems like the best we can do.

    @param text (str): a "word" (may contain punctuation)
    @return a list of processed strings with punctuation and stop words removed
    """

    # Words to omit
    stop_list = stopwords.words('english')
    omit = set(stop_list).union(set(houzz.LABELS))  # words we don't use

    # Replace punctuation by ' '
    for p in string.punctuation:
        text = text.replace(p, ' ')

    return [word.lower() for word in text.split() if word not in omit]


def image_features(txt_file, img_dir, output_dir, feature_of):
    """
    Compute and save feature representations for the images
    in the dataset.

    @param txt_file (str): 
               path of a text file listing the images (.jpg)
               in the dataset in the CaffeNet style 
    @param img_dir (str):
                path of the directory containing the images (.jpg)
    @param output_dir (str): 
                path where you want the output files (.npy) to go
    @param feature_of (func: str -> ndarray): 
               function that computes the feature representation
               of an image (.jpg)
    """
    img_dir = standardize(img_dir)
    output_dir = standardize(output_dir)
    with open(txt_file, 'r') as dataset:
        for line in dataset:
            img_file = line.split()[0]
            feature = feature_of(img_dir + img_file)
            npy = img_file.replace('.jpg', '.npy')
            np.save(output_dir + npy, feature)

            format_print("Output written for {}".format(img_file))


def caffenet_features(text_file, lmdb_folder, output_folder):
    """
    Load precomputed CaffeNet features from the LMDB database
    and save them as numpy arrays.
    """
    lmdb_env = lmdb.open(lmdb_folder)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe_pb2.Datum()

    with open(text_file, 'r') as f:

        for key, value in lmdb_cursor:
            datum.ParseFromString(value)

            # Load the extracted feature
            data = caffe.io.datum_to_array(datum)
            # Filename with .npy extension
            filename = f.readline().split()[0]
            output_file = output_folder + filename[:-len('.jpg')] + '.npy'

            np.save(output_file, data)


def gist_feature(img_path):
    """
    Compute the GIST feature for a JPG image.

    @param img_path (str): location of an image (.jpg)
    @return ndarray for a GIST descriptor
    """
    im = Image.open(img_path)
    return leargist.color_gist(im)


def hsv_gist_feature(img_path):
    """
    Concatenate an HSV histogram and GIST feature.

    Precondition:
    HSV and GIST features precomputed.

    To use in image_features, this function may take only one parameter,
    the location of the image, even though it relies on the locations
    of the precomputed features.
    Set GIST_DIR and HIST_DIR to tell this function where to look
    for the precomputed features.
    """
    GIST_DIR = fullfile(houzz.DATASET_ROOT, 'img/GIST/')
    HIST_DIR = fullfile(houzz.DATASET_ROOT, 'img/HSVH/')

    # Get the name of the data instance
    relative_pathname = img_path.split('/')[-1]
    name = relative_pathname[:-len('.jpg')] + '.npy'

    gist = np.load(GIST_DIR + name)
    hist = np.load(HIST_DIR + name)
    return np.concatenate((gist, hist))


# Standard RGB max values (used by scipy.misc.imread())
STD_MAX_R = 255.0
STD_MAX_G = 255.0
STD_MAX_B = 255.0

# HSV max values, according to colorsys
MAX_H, MAX_S, MAX_V = 1.0, 1.0, 1.0


class HSVHistogram(object):
    """
    HSVHistogram
    Data structure for HSV histograms

    Methods:
        - bin(self, h, s, v):
            add HSV value to the histogram
        - count(self, h_idx, s_idx, v_idx):
            get frequency from a bin
        - as_list()
            return the histogram contents as a vector
    """
    # Class constants
    NUM_BINS = 10
    H_BIN_SIZE = MAX_H/NUM_BINS
    S_BIN_SIZE = MAX_S/NUM_BINS
    V_BIN_SIZE = MAX_V/NUM_BINS

    def __init__(self):
        self.total = 0  # total number of entries; used to normaliize
        # Creates a 3-D double array w/ each entry init to 0.0
        size = (self.NUM_BINS, self.NUM_BINS, self.NUM_BINS)
        self.hist = np.zeros(size)

    def bin(self, h, s, v):
        """
        @param HSV value
        Side Effect: bins value and updates histogram
        """
        # Bin number = floor(val/BIN_SIZE)
        # Max value -> last bin (formula causes out of bounds)
        h_idx = self.NUM_BINS-1 if approx_equals(h, MAX_H) else int(h/self.H_BIN_SIZE) 
        s_idx = self.NUM_BINS-1 if approx_equals(s, MAX_S) else int(s/self.S_BIN_SIZE)
        v_idx = self.NUM_BINS-1 if approx_equals(v, MAX_V) else int(v/self.V_BIN_SIZE)
         
        # Add to bin count
        self.hist[h_idx, s_idx, v_idx] += 1
        self.total += 1

    def count(self, h_idx, s_idx, v_idx):
        """
        @param bin coordinate in {0, 1, ..., NUM_BINS}
        @return normalized count
        """
        if self.total == 0:
            return 0
        else:
            return self.hist[h_idx, s_idx, v_idx]/self.total
            # hist is an ndarray of floats

    def as_list(self):
        """
        @return (list): histogram of color frequencies
        """
        to_return = []
        for i in xrange(0, self.NUM_BINS):
            for j in xrange(0, self.NUM_BINS):
                for k in xrange(0, self.NUM_BINS):
                    to_return.append(self.count(i, j, k))

        return to_return


def to_rgb(im):
    # This should be fsater than 1, as we only
    # truncate to uint8 once (?)
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = im
    return ret


def hsv_hist(im_filename):
    """
    Compute a color histogram representation of an image.

    @param im_filename (str): image filename
    @return (HSVHistogram): 10-bin HSV color histogram
    @return None if given a grayscale image
    """
    I = imread(im_filename)   # image as a numpy ndarray
    # Check grayscale
    if len(I.shape) != 3:
        I = to_rgb(I)

    hist = HSVHistogram()     # create a new histogram

    # Bin each pixel in the image
    for row in I:
        for pixel in row:
            r, g, b = pixel
            # Scale to range [0.0, 1.0] expected by colorsys
            r /= STD_MAX_R
            g /= STD_MAX_G
            b /= STD_MAX_B
            h, s, v = rgb_to_hsv(r, g, b)

            hist.bin(h, s, v)

    print("Processed " + im_filename)
    return hist


def approx_equals(x, y):
    """
    Check floating-point "equality," up to some tolerance.

    @param two floating point values, x and y
    @return True or False
    """
    tol = 10**(-5)
    return True if abs(x - y) < tol else False
