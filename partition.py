"""
Script to partition the dataset into training and test sets.
"""

import os
import cPickle as pickle

from houzz import DATASET_ROOT

# Get a list of filenames for pickled metadata
PICKLED_PATH = DATASET_ROOT + 'metadata/bedroom/pkl/'
pickles = os.listdir(PICKLED_PATH)
# Two of these files do not actually contain metadata ...
pickles.remove('successes.pkl')
pickles.remove('failures.pkl')

"""
Keep images with text data (description or tag).
"""

keep = []

for pkl in pickles:
    with open(PICKLED_PATH + pkl) as fd:
        data = pickle.load(fd)  # a dictionary 
        if data['tag'] or data['description']:
            keep.append(pkl.rstrip('.pkl'))
            # We'll use this name to find the corresponding image

"""
Go through the images and text data for the items we have kept
and load their features.

Precondition: feature representations have been computed for images
and text.
"""
