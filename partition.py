"""
Script to partition the dataset into training and test sets.
"""

import os
import cPickle as pickle

# Change this if running on a different machine (end with '/')
DATASET_ROOT = '/home/brian/LanguageVision/final_project/dataset/'

# Get a list of filenames for pickled metadata
PICKLED_PATH = DATASET_ROOT + 'metadata/bedroom/pkl/'
pickles = os.listdir(PICKLED_PATH)
# Two of these files do not actually contain metadata ...
pickles.remove('successes.pkl')
pickles.remove('failures.pkl')

"""
Count how many files have tags and descriptions.
"""
with_tags  = set()
with_descr = set()

for pkl in pickles:
    with open(PICKLED_PATH + pkl) as fd:
        data = pickle.load(fd)  # a dictionary 
        if data['tag']:
            with_tags.add(pkl)
        if data['description']:
            with_descr.add(pkl)


"""
Results
"""
print "Number with tags: {}".format(len(with_tags))
print "Number with descriptions: {}".format(len(with_descr))
print "Number with both: {}".format(len(with_tags.intersection(with_descr)))
print "Number with neither: {}".format(len(pickles) - len(with_tags.union(with_descr)))
