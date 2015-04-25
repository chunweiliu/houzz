import sys
import os
import cPickle

import liblinearutil

from learning import train_svm
from learning import confusion
from learning import load_dataset
from houzz import partition
from format_print import format_print
import plot_confusion_matrix
# from houzz import balance_partition as partition


def main(load_img, load_txt, n_test, n_train, pca_k):
    # Training
    model_name = 'img_{0}_txt_{1}_test_{2}_train_{3}_pcak_{4}'.format(
        load_img, load_txt, n_test, n_train, pca_k)
    img_feature_dir = '/home/chunwei/Data/houzz/caffenet_features'
    txt_feature_dir = '/home/chunwei/Data/houzz/text_features'
    model_root = '/home/chunwei/Data/houzz/trained_svms'
    model_file = model_root + '/' + model_name + '.model'
    model_meta = model_root + '/' + model_name + '.pkl'
    vis_file = model_root + '/' + model_name + '.png'

    format_print("Routine start ...")
    if not os.path.exists(model_file) or not os.path.exists(model_meta):
        (train_labels, test_labels) = partition(n_test, n_train)
        model, img_sf, txt_sf, pca = \
            train_svm(model_name, train_labels,
                      img_feature_dir, txt_feature_dir, model_root,
                      load_img, load_txt, pca_k)
        # Save the trained model and its meta
        liblinearutil.save_model(model_file, model)
        with open(model_meta, 'w') as f:
            # from the smallest to the largest
            cPickle.dump(img_sf, f)
            cPickle.dump(txt_sf, f)
            cPickle.dump(pca, f)
            cPickle.dump(test_labels, f)
            cPickle.dump(train_labels, f)
    else:
        model = liblinearutil.load_model(model_file)
        with open(model_meta, 'r') as f:
            img_sf = cPickle.load(f)
            txt_sf = cPickle.load(f)
            pca = cPickle.load(f)
            test_labels = cPickle.load(f)
            train_labels = cPickle.load(f)

    # Testing
    format_print("Testing ...")
    x, y, _, _ = load_dataset(test_labels, img_feature_dir, txt_feature_dir,
                              load_img, load_txt, img_sf, txt_sf)

    # PCA
    x = pca.project(x)
    x = x[:, :pca.K]  # data x projected on the priciple plane

    x = x.tolist()
    y = y.tolist()
    predicted_labels, stats, p_val = liblinearutil.predict(y, x, model, '-q')

    # Visualize the result
    format_print("Results ...")
    mat = confusion(y, predicted_labels)
    mat *= 100
    accuracy = stats[0]
    print mat
    print accuracy

    plot_confusion_matrix.plot(mat, vis_file)


if __name__ == '__main__':
    load_img = int(sys.argv[1])
    load_txt = int(sys.argv[2])
    n_test = int(sys.argv[3])
    n_train = int(sys.argv[4])
    pca_k = int(sys.argv[5])
    main(load_img, load_txt, n_test, n_train, pca_k)
