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
    return scale(text_vector)


def scale(vector):
    """
    Map each element in the feature to the range [-1, 1].

    @param vector (ndarray)
    @returns (ndarray) scaled feature
    """
    # Divide by the element with the largest absolute value
    normalizer = max(abs(vector.max()), abs(vector.min()))
    return vector / normalizer if normalizer else vector


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

            # Load the extracted feature and scale to [-1, 1]
            data = scale(caffe.io.datum_to_array(datum))
            # Filename with .npy extension
            filename = f.readline().split()[0]
            output_file = output_folder + filename.rstrip('.jpg') + '.npy'

            np.save(output_file, data)


def feature(filename, img_dir, txt_dir=houzz.DATASET_ROOT + 'text_features'):
    """
    Compute the combined feature for the data item.

    Preconditions:
        1) Image features precomputed and stored in img_dir
        2) Text features precomputed and stored in txt_dir

    @param filename (str): data_xxxx
    @param img_dir (str): location of image features
    @param txt_dir (str): location of text features

    @return (list) combined img + text feature representation
    """
    img_dir = houzz.standardize(img_dir)
    txt_dir = houzz.standardize(txt_dir)

    # Load features from .npy files
    img = np.load(img_dir + filename + '.npy')
    txt = np.load(txt_dir + filename + '.npy')

    # Concatenate
    return np.concatenate((img.flatten(), txt.flatten())).tolist()
