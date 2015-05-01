"""
Experiment module for computing the results
        NONE    DESC    TAGS    DESC + TAGS
NONE
GIST
HSVH
HSVG
CAFE
"""

import os

import liblinearutil

from learning import train_svm
from utilities import fullfile


def train(img_feature_type, txt_feature_type, feature_dir, pca_k,
          n_test, n_train, svm_dir):
    """
    Given the type of feautres, train a svm
    @param (str) img_feature_type
    @param (str) txt_feature_type
    @param (str) feature_dir
    @param (str) pca_k, the number of principle component returns in PCA
    @param (str) n_train, number of training data
    @param (str) n_test, number of test data
    @param (str) svm_dir, the place to store the pkl and model files
    """
    model_path, model_meta = pack_name(img_feature_type, txt_feature_type,
                                       pca_k, n_test, n_train, svm_dir)

    # Train a model if there are no such model and meta existed
    if not os.path.exists(model_path) and not os.path.exists(model_meta):
        # Get the partition
        (train_labels, test_labels) = partition(n_test, n_train)

        # Get the image and text feature paths
        img_feature_dir = fullfile(feature_dir, img_feature_type) + '/'
        txt_feature_dir = fullfile(feature_dir, txt_feature_type) + '/'

        # Train the SVM (need to refactorize)
        model, scale_factors, pca = train_svm(train_labels, img_feature_dir,
                                              txt_feature_dir, pca_k)
        # model, img_sf, txt_sf, pca = \
        #     train_svm(model_name, train_labels,
        #               img_feature_dir, txt_feature_dir, model_root,
        #               load_img, load_txt, pca_k)

        # Save the trained model and its meta
        liblinearutil.save_model(model_file, model)
        with open(model_meta, 'w') as f:
            # from the smallest to the largest
            cPickle.dump(scale_factors.img, f)
            cPickle.dump(scale_factors.txt, f)
            cPickle.dump(pca, f)
            cPickle.dump(test_labels, f)
            cPickle.dump(train_labels, f)
    else:
        pass


def test(img_feature_type, txt_feature_type, feature_dir, pca_k,
         n_test, n_train, svm_dir):
    """
    Test pretrained SVM
    @returns (array) mat, a normalized confusion matrix
    @returns (float) acc, an accuray of the test labels
    """
    model_path, model_meta = pack_name(img_feature_type, txt_feature_type,
                                       pca_k, n_test, n_train, svm_dir)
    if not os.path.exists(model_path) or not os.past.exists(model_meta):
        train(img_feature_type, txt_feature_type, feature_dir, pca_k,
              n_test, n_train, svm_dir)

    model, meta = load_model(model_meta, model_meta)

    # Get the image and text feature paths
    img_feature_dir = fullfile(feature_dir, img_feature_type) + '/'
    txt_feature_dir = fullfile(feature_dir, txt_feature_type) + '/'

    mat, acc = test_svm(test_labels, img_feature_dir, txt_feature_dir,
                        model, scale_factors, pca)
    return mat, acc


def pack_name(img_feature_type, txt_feature_type, pca_k, n_test, n_train,
              output_dir):
    """
    Pack the model's name and meta from the input arguments
    """
    # Specify the model path
    model_name = 'img_{0}_txt_{1}_pca_{2}_test_{3}_train_{4}.model'.format(
        img_feature_type, txt_feature_type, pca_k, n_test, n_train)
    model_path = fullfile(output_dir, model_name)
    # Specify the model meta path
    model_meta = fullfile(output_dir, model_name[:-len('.model')] + '.pkl')
    return model_name, model_meta


def load_model(model_path, model_meta):
    model = liblinearutil.load_model(model_file)
    with open(model_meta, 'r') as f:
        scale_factors.img = cPickle.load(f)
        scale_factors.txt = cPickle.load(f)
        pca = cPickle.load(f)
        test_labels = cPickle.load(f)
        train_labels = cPickle.load(f)
    meta = (scale_factors, pca, test_labels, train_labels)
    return model, meta

def run():
