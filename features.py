"""
Subroutines for working with the various feature representations.
"""

import os
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
    """Precompute all textual features for the files in the meta_folder
    """
	
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
            # make a check
            try:
                if feature.any() is not False:
                    npy = pkl.rstrip('.pkl') + '.npy'
                    np.save(text_folder + npy, feature)
            except AttributeError:
                pass


def compute_text_feature(metadata, model):
    """Compute a mean vector to represent the words in description and tags

    @param metadata
        The metadata from self.loadmat
    """

    if not metadata['tag'] and not metadata['description']:
        format_print('No textual feature found')
        return None

    tag_feature = np.zeros(300, dtype=np.float32)
    if metadata['tag']:
        tag_feature += average_text_vector(metadata['tag'], model)

    des_feature = np.zeros(300, dtype=np.float32)
    if metadata['description']:
        des_feature += average_text_vector(
            metadata['description'].split(), model)

    feature = tag_feature + des_feature  # merge two with equal weights
    norm = np.linalg.norm(feature)
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
    feature = np.zeros(300, dtype=np.float32)
    for word in words:
        processed_words = process_text(word)
        for processed_word in processed_words:
            if processed_word in model:
                feature += model[processed_word]
    # return a normalized feature
    norm = np.linalg.norm(feature)
    return feature / norm if norm else feature


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

            # label = datum.label
            data = caffe.io.datum_to_array(datum)
            filename = f.readline().split()[0]
            output_file = output_folder + filename.rstrip('.jpg') + '.npy'
            np.save(output_file, data)


def feature(filename, img_dir, txt_dir=houzz.DATASET_ROOT + 'text_features'):
	"""
	Preconditions:
		1) Image features precomputed and stored in img_dir
		2) Text features precomputed and stored in txt_dir
	@param
		filename (str): data_xxxx
		img_dir (str): location of image features 
		txt_dir (str): location of text features 

	@returns
		Combined img + text feature representation
	"""
	img_dir = houzz.standardize(img_dir)
	txt_dir = houzz.standardize(txt_dir)

	# Load features from .npy files
	img = np.load(img_dir + filename + '.npy')
	txt = np.load(txt_dir + filename + '.npy')
	
	# Concatenate
	return np.concatenate( (img.flatten(), txt.flatten()) )
