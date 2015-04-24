import sys
import os.path
import cPickle

import liblinearutil

from learning import train_svm
from learning import confusion
from learning import load_dataset
from houzz import balance_partition


def main(load_img, load_txt, n_test, n_train):
    # Training
    model_name = 'img_{0}_txt_{1}_test_{2}_train_{3}'.format(
        load_img, load_txt, n_test, n_train)
    img_feature_dir = '/home/chunwei/Data/houzz/caffenet_features'
    txt_feature_dir = '/home/chunwei/Data/houzz/text_features'
    model_root = '/home/chunwei/Data/houzz/trained_svms'
    # model_file = model_root + '/' + model_name + '.pkl'

    # (train_labels, test_labels) = partition(test_fraction=10, stop=stop)
    (train_labels, test_labels) = balance_partition(n_test, n_train)
    model, img_sf, txt_sf, pca = \
        train_svm(model_name, train_labels,
                  img_feature_dir, txt_feature_dir, model_root,
                  load_img, load_txt)

    # Testing
    x, y, _, _ = load_dataset(test_labels, img_feature_dir, txt_feature_dir,
                              load_img, load_txt, img_sf, txt_sf)

    # PCA
    x = pca.project(x)
    x = x[:, :pca.K]  # data x projected on the priciple plane
    y = y[:pca.K]

    x = x.tolist()
    y = y.tolist()
    predicted_labels, stats, p_val = liblinearutil.predict(y, x, model, '-q')

    accuracy = stats[0]

    # Visualize the result
    mat = confusion(y, predicted_labels)

    print mat
    print accuracy


if __name__ == '__main__':
    load_img = int(sys.argv[1])
    load_txt = int(sys.argv[2])
    n_test = int(sys.argv[3])
    n_train = int(sys.argv[4])
    main(load_img, load_txt, n_test, n_train)
