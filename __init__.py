import scipy.io
import numpy

import gensim
from nltk.corpus import stopwords

from format_print import format_print

# Installation: set this to the correct path
# End with trailing '/'
DATASET_ROOT = '/home/brian/LanguageVision/final_project/dataset/'

# List of labels in our closed world
LABELS = ["traditional", 
          "contemporary", 
          "eclectic",
          "modern",
          "mediterranean",
          "tropical",
          "asian"
         ]

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
        """ Assume Unix-style pathnames
        Remove trailing '/' from 'base_folder,' if present, to regularize input
        """
        base_folder = base_folder.rstrip('/')

        # Set instance variables for paths
        self.base_folder = base_folder
        self.image_folder = base_folder + '/images/' + category
        self.keys_file = base_folder + '/keys/' + category + '.mat'
        self.data_folder = base_folder + '/metadata/' + category
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

    def compute_text_feature(self, filename):
        """Compute a mean vector to represent the words in description and tags

        @param metadata
            The metadata from self.loadmat
        """
        metadata = self.loadmat(filename)

        if not metadata['tag'] and not metadata['description']:
            format_print('No textual feature found')
            return None

        if metadata['tag']:
            tag_feature = self.average_text_vector(metadata['tag'])

        if metadata['description']:
            des_feature = self.average_text_vector(
                metadata['description'].split())

        return tag_feature, des_feature

    def process_text(self, text):
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
        omit = set(stop_list).union(set(LABELS))  # words we don't want to use

        return [word for word in text.split('-') if word not in omit]

    def average_text_vector(self, words):
        """
        Compute a feature vector for one term. If the term is a phrase joined
        by hypehns, the function split it to words and average the individual
        words as one vector.

        @param words
            a list of (possibly hyphenated) words
        @returns
            a normalized vector
        """
        feature = numpy.array([0] * self.N_TEXT_FEATURE, dtype=numpy.float32)
        for word in words:
            processed_words = self.process_text(word)
            for processed_word in processed_words:
                if processed_word in self.text_model:
                    feature += self.text_model[processed_word]
        # return a normalized feature
        norm = numpy.linalg.norm(feature)
        return feature / norm if norm else feature

    def check_path(self, filename):
        """
        Check the filename, prepend or append information if needs

        @ param filename
            a string
        @ returns
            a processed filename
        """
        if filename[0] == '/':
            return filename
        else: 
            # need to append the root folder
            return self.mat_data_folder + '/' + filename


"""
We have made a closed world assumption that there are only seven
possible style labels (see the LABELS list in this module)
because many labels only appear one or two times
or are not actual style descriptions.

We want a list of only those items that we will use for our task,
along with their label numbers. This file will be in the format
expected by Caffe. When we need to use it in Python,
we will parse it into a dictionary using the dataset_dict() function
in this module.
"""
import os
data_file = "bedroom.txt"

if data_file not in os.listdir('.'):
    # Generate this file
    dataset = Houzz()
    with open(data_file, 'w') as fd:
        for mat in os.listdir(dataset.mat_data_folder):
            data = dataset.loadmat(mat)
            # Check if it has a label we want 
            if data['style'] in LABELS and (data['tag'] or data['description']):
                # Write a line with 
                #   <name>.jpg <label number> 
                # to the file
                jpg = mat.rstrip('.mat') + '.jpg'
                label = LABELS.index(data['style'])
                fd.write(jpg + ' ' + str(label) + '\n')


#def dataset_dict():