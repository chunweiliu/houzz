"""
A script to turn the dataset's .mat metadata files into .pkl files
that Python can work with more handily.

This script assumes that the dataset has a directory hierarchy that looks like:

    root
        metadata
            bedroom
                mat
                pkl

Brian Cristante
30 March 2015
"""

import os
import cPickle as pickle
from Houzz import Houzz

DATASET_ROOT = "/home/brian/LanguageVision/final_project/dataset/"

# Create Houzz object for the dataset
dataset = Houzz(DATASET_ROOT, "bedroom")

# 'mat' and 'pkl' directories for metadata
mat_data_dir = dataset.data_folder + "/mat/"
pkl_data_dir = dataset.data_folder + "/pkl/"

# Loop to create pkl files from mat files
mat_data = os.listdir(mat_data_dir)
successes = []
failures  = []

for mat in mat_data:
    # Process the mat file into a Python dictionary
    print mat_data_dir + mat
    
    # Not all files have all fields, notably 'tag,'
    # so this operation may raise a ValueError or IndexError
    try:
        meta = dataset.loadmat(mat_data_dir + mat)
    except Exception:
        failures.append(mat)
        continue

    # Filename 
    name = mat.rstrip(".mat")                 # 'mat' is a relative path
    dump_name = pkl_data_dir + name + ".pkl"  # absolute path
    # Pickle
    with open(dump_name, 'wb') as pkl:
        pickle.dump(meta, pkl)
    print(mat + " pickled")
    successes.append(mat)

# Pickle lists of successes and failures
with open(pkl_data_dir + "successes.pkl", 'wb') as fd:
    pickle.dump(successes, fd)

with open(pkl_data_dir + "failures.pkl", 'wb') as fd:
    pickle.dump(failures, fd)

print("\n\n\nPickling complete!\n")
