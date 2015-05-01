import collections

import scipy.io
import gensim

from utilities import format_print

# Installation: set this to the correct path
# End with trailing '/'
DATASET_ROOT = '/home/chunwei/Data/houzz/'
# Where the program will look for and save trained SVMs
TRAINED_PATH = DATASET_ROOT + 'trained_svms'

# List of labels in our closed world
LABELS = ["traditional",
          "contemporary",
          "eclectic",
          "modern",
          "mediterranean",
          "tropical",
          "asian"]


def standardize(path):
    """
    Adds trailing '/' if absent.
    @param path (str): Unix-style path
    @return std (str): Unix-style path with trailing '/'
    """
    return path if path[-1] == '/' else path + '/'


class Houzz:
    """A class for processing the Houzz.com dataset:
    Houzz
        - images
        - keys
        - metadata
            pkl
            meta

    A single instance of this class represents the entire dataset.
    The 'loadmat()' method creates a Python dictionary
    for a single metadata file.
    """
    # Define class constants
    N_TEXT_FEATURE = 300

    def __init__(self, base_folder=DATASET_ROOT, category='bedroom',
                 text_model_file='GoogleNews-vectors-negative300.bin'):
        base_folder = standardize(base_folder)

        # Set instance variables for paths
        self.base_folder = base_folder
        self.image_folder = base_folder + 'images/' + category
        self.keys_file = base_folder + 'keys/' + category + '.mat'
        self.data_folder = base_folder + 'metadata/' + category
        self.mat_data_folder = self.data_folder + '/mat'
        self.pkl_data_folder = self.data_folder + '/pkl'
        self.text_model_file = text_model_file

    def build(self):
        """Perform time consuming tasks
        """
        # Set instance variables for the text model
        format_print('Loading the Google News Model ...')
        self.text_model = gensim.models.Word2Vec.load_word2vec_format(
            self.text_model_file, binary=True)  # the Google News Model
        format_print('... Finish loading')

        # Seperate training and testing data

    def loadmat(self, filename):
        """An example of the Matlab format of the Houzz dataset (data_02101038)
             data =
                 url: [1x83 char]
                  id: 2101038
          image_link: [1x74 char]
         description: [1x134 char]
               style: 'contemporary'
                type: 'bedroom'
                 tag: {'built-in-desk'  'guest-room-retreat'  'vaulted-ceilings'}
            sameproj: {[1x77 char]  [1x83 char]  [1x87 char]}
           recommend: {1x6 cell}
        """
        filename = self.check_path(filename)  # append the root if needed

        py_mat = scipy.io.loadmat(filename, squeeze_me=True,
                                  struct_as_record=False)
        py_data = py_mat['data']
        attributes = dir(py_data)

        data = dict()

        # for string array in matlab
        data['url'] = py_data.url                          if 'url' in attributes else None
        data['id'] = py_data.id                            if 'id' in attributes else None
        data['image_link'] = py_data.image_link            if 'image_link' in attributes else None
        data['description'] = py_data.description          if 'description' in attributes else None
        data['style'] = py_data.style                      if 'style' in attributes else None
        data['type'] = py_data.type                        if 'type' in attributes else None

        # for cell array in Matlab
        data['tag'] = [x for x in py_data.tag]             if 'tag' in attributes else None
        data['sameproj'] = [x for x in py_data.sameproj]   if 'sameproj' in attributes else None
        data['recommend'] = [x for x in py_data.recommend] if 'recommend' in attributes else None
        return data

    def check_path(self, filename):
        """
        Check the filename, prepend or append information if needs

        @ param filename: a string
        @ return a processed filename
        """
        if filename[0] == '/':
            return filename
        else:
            # need to append the root folder
            return self.mat_data_folder + '/' + filename


def partition(n_test, n_train, file_list='bedroom.txt'):
    """
    Partitions the dataset into training and test sets.
    @param n_test (int)
        number of test data.
    @param n_train (int)
        number of training data.
    @param file_list (str)
        a text file in the format expected by Caffe.
        <xxxx.jpg> <int>
    """

    train = dict()
    test = dict()
    with open(file_list) as f:
        # start with test data
        for i, line in enumerate(f):
            if i < n_test:
                filename = line.split()[0].rstrip('.jpg')
                label = int(line.split()[1])
                test[filename] = label
            elif i < n_test + n_train:
                filename = line.split()[0].rstrip('.jpg')
                label = int(line.split()[1])
                train[filename] = label
            else:
                break
    return (train, test)


def balance_partition(n_test, n_train, file_list='bedroom.txt'):
    """
    Partitions the datatset into training and test sets.
    The test set has balance numbers of sample in each class.

    @param n_test (int)
        number of test data

    @param n_train (int)
        number of training data

    @param file_list
        a text file in the format expected by Caffe.
        <xxxx.jpg> <int>
    """
    # Save each file in different bins
    bins = collections.defaultdict(list)
    with open(file_list) as f:
        for line in f:
            filename = line.split()[0].rstrip('.jpg')
            label = int(line.split()[1])
            bins[LABELS[label]].append(filename)

    # Instantiate iterators for each bins
    iterbins = dict()
    for key in bins:
        iterbins[key] = iter(bins[key])

    test = dict()
    for i in range(n_test):
        label = LABELS[i % len(LABELS)]
        next_item = next(iterbins[label], None)
        if next_item:
            test[next_item] = i % len(LABELS)

    train = dict()
    for i in range(n_test, n_test + n_train):
        label = LABELS[i % len(LABELS)]
        next_item = next(iterbins[label], None)
        if next_item:
            train[next_item] = i % len(LABELS)

    return (train, test)

"""
TODO probably not a good idea
Severely restricts portability of this module

When this module is loaded, ensure that 'bedroom.txt' is
in the current directory.

bedroom.txt is a list of only those items that we will use for our task,
along with their label numbers. This file will be in the format
expected by Caffe.

import os
data_file = "bedroom.txt"

if data_file not in os.listdir('.'):
    # Generate this file
    dataset = Houzz()
    with open(data_file, 'w') as fd:
        for mat in os.listdir(dataset.mat_data_folder):
            data = dataset.loadmat(mat)
            # Check if it has a label we want
            if data['style'] in LABELS and \
               (data['tag'] or data['description']):
                # Write a line with
                #   <name>.jpg <label number>
                # to the file
                jpg = mat.rstrip('.mat') + '.jpg'
                label = LABELS.index(data['style'])
                fd.write(jpg + ' ' + str(label) + '\n')
"""
