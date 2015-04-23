import sys

import liblinearutil

from learning import train_svm
from learning import confusion
from learning import load_dataset
from learning import scale
from houzz import balance_partition


def main(load_img, load_txt, n_test, n_train):
    # Training
    model_name = 'img_{0}_txt_{1}_test_{2}_train_{3}'.format(
        load_img, load_txt, n_test, n_train)
    img_feature_dir = '/home/chunwei/Data/houzz/caffenet_features'
    txt_feature_dir = '/home/chunwei/Data/houzz/text_features'
    output_dir = '/home/chunwei/Data/houzz/trained_svms'

    # (train_labels, test_labels) = partition(test_fraction=10, stop=stop)
    (train_labels, test_labels) = balance_partition(n_test, n_train)
    _, scale_factor = train_svm(model_name, train_labels,
                                img_feature_dir, txt_feature_dir,
                                output_dir,
                                load_img, load_txt)

    # Testing
    model_root = output_dir
    model_file = model_root + '/' +model_name + '.model'
    model = liblinearutil.load_model(model_file)

    x, y = load_dataset(test_labels, img_feature_dir, txt_feature_dir,
                        load_img, load_txt)
    if load_img and load_txt:
        x = scale(x, scale_factor)

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
