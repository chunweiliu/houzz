"""
Subroutines for working with the various feature representations.
"""

import os
import string
import cPickle as pickle
import numpy as np

import gensim
from nltk.corpus import stopwords

from format_print import format_print
import houzz

import caffe
from caffe.proto import caffe_pb2
import lmdb


def text_feature(meta_folder, text_model_file, text_folder):
    """Precompute all textual features for the files in the meta_folder."""

    meta_folder = houzz.standardize(meta_folder)
    text_folder = houzz.standardize(text_folder)

    # Load the langauge model
    format_print('Loading the Google News Model ...')
    model = gensim.models.Word2Vec.load_word2vec_format(
        text_model_file, binary=True)  # the Google News Model
    format_print('... Finish loading')

    for pkl in os.listdir(meta_folder):
        with open(meta_folder + pkl, 'r') as f:
            metadata = pickle.load(f)
            feature = compute_text_feature(metadata, model)
            # Discard empty features
            try:
                if feature.any() is not False:
                    npy = pkl.rstrip('.pkl') + '.npy'
                    np.save(text_folder + npy, feature)
            except AttributeError:
                pass


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


# def scale(vector):
#     """
#     Map each element in the feature to the range [-1, 1].

#     @param vector (ndarray)
#     @returns (ndarray) scaled feature
#     """
#     # Divide by the element with the largest absolute value
#     normalizer = max(abs(vector.max()), abs(vector.min()))
#     return vector / normalizer if normalizer else vector


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


def image_feature(text_file, lmdb_folder, output_folder):
    """Parse the file names and labels in the text_file, search for the
    corresponding features in the lmdb database, write an individual file
    for each record in the output_folder
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
            output_file = output_folder + filename.rstrip('.jpg') + '.npy'

            np.save(output_file, data)


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


def load_dataset(names_to_labels, img_dir, txt_dir,
                 load_img=True, load_txt=True, img_sf=None, txt_sf=None):
    """
    Combine individually scaled image and text features for the dataset.

    Preconditions:
        1) Image features precomputed and stored in img_dir
        2) Text features precomputed and stored in txt_dir

    @param names_to_labels (dict: str -> int)
    @param img_dir (str)
    @param txt_dir (str)
    @param load_img (bool): set to use image features
    @param load_txt (bool): set to use text features

    @return list of features
    @return list of labels (same order as list of features)
    @return (float): image feature's scale factor
    @return (float): text feature's scale factor
    """

    img_dir = houzz.standardize(img_dir)
    txt_dir = houzz.standardize(txt_dir)

    img_features = []
    txt_features = []

    # Load all image and text features to compute scaling
    names = names_to_labels.keys()
    for stem in names:
        if load_img:
            img = np.load(img_dir + stem + '.npy').flatten()
        else:
            img = np.array([])

        if load_txt:
            txt = np.load(txt_dir + stem + '.npy').flatten()
        else:
            txt = np.array([])

        img_features.append(img)
        txt_features.append(txt)

    # Turn lists to ndarrays for scaling
    img_features = np.array(img_features)
    txt_features = np.array(txt_features)

    # Scale image and text features separately
    if not img_sf:
        img_sf = find_scale_factor(img_features)
    if not txt_sf:
        txt_sf = find_scale_factor(txt_features)

    img_features = scale(img_features, img_sf)
    txt_features = scale(txt_features, txt_sf)

    # Concatenate scaled features
    x, y = [], []
    for img, txt, stem in zip(img_features, txt_features, names):
        x.append(np.concatenate((img, txt)))
        y.append(names_to_labels[stem])

    return np.array(x), np.array(y), img_sf, txt_sf
