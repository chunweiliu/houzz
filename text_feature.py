import os
import cPickle as pickle
import numpy

import gensim
from nltk.corpus import stopwords

from format_print import format_print
import houzz


def text_feature(meta_folder, text_model_file, text_folder):
    """Precompute all textual features for the files in the meta_folder
    """
    # Unified the format of folder
    if meta_folder[-1] != '/':
        meta_folder += '/'
    if text_folder[-1] != '/':
        text_folder += '/'

    # Load the langauge model
    format_print('Loading the Google News Model ...')
    model = gensim.models.Word2Vec.load_word2vec_format(
        text_model_file, binary=True)  # the Google News Model
    format_print('... Finish loading')

    for pkl in os.listdir(meta_folder):
        with open(meta_folder + pkl, 'r') as f:
            metadata = pickle.load(f)
            feature = compute_text_feature(metadata, model)

            filename = pkl.rstrip('.pkl') + '.npy'
            numpy.save(text_folder + filename, feature)


def compute_text_feature(metadata, model):
    """Compute a mean vector to represent the words in description and tags

    @param metadata
        The metadata from self.loadmat
    """

    if not metadata['tag'] and not metadata['description']:
        format_print('No textual feature found')
        return None

    tag_feature = numpy.zeros(300, dtype=numpy.float32)
    if metadata['tag']:
        tag_feature += average_text_vector(metadata['tag'], model)

    des_feature = numpy.zeros(300, dtype=numpy.float32)
    if metadata['description']:
        des_feature += average_text_vector(
            metadata['description'].split(), model)

    feature = tag_feature + des_feature  # merge two with equal weights
    norm = numpy.linalg.norm(feature)
    return feature / norm if norm else feature


def process_text(text):
    """
    Each tag in the dataset is a phrase joined by hyphens. For example,

        guest-room-retreat

    This will not appear in the word2vec lexicon. Therefore,
    split each tag into its constituent words and remove stop words.
    Without a special corpus of interior design terms to train the
    language model on, this seems like the best we can do.

    @param text
        a (possibly hyphenated) tag
    @returns
        a list of processed strings with hyphens and stop words removed
    """
    stop_list = stopwords.words('english')
    omit = set(stop_list).union(set(houzz.LABELS))  # words we don't use

    return [word for word in text.split('-') if word not in omit]


def average_text_vector(words, model):
    """
    Compute a feature vector for one term. If the term is a phrase joined
    by hypehns, the function split it to words and average the individual
    words as one vector.

    @param words
        a list of (possibly hyphenated) words
    @returns
        a normalized vector
    """
    feature = numpy.zeros(300, dtype=numpy.float32)
    for word in words:
        processed_words = process_text(word)
        for processed_word in processed_words:
            if processed_word in model:
                feature += model[processed_word]
    # return a normalized feature
    norm = numpy.linalg.norm(feature)
    return feature / norm if norm else feature
