"""
Experiment module for computing the results
        NONE    DESC    TAGS    DESC + TAGS
NONE
GIST
HSVH
HSVG
CAFFE
"""

import os

import houzz
import utilities
import learning

# Determine locations of image features
IMG_FEATURE_TYPES = ['CAFFE']

# Determine locations of text features
TXT_FEATURE_TYPES = ['TAGS', 'BOTH']


def run():
    name = '/home/chunwei/Projects/houzz/bedroom.txt'
    if not os.path.exists(name):
        houzz.create_data_file(name)

    n_train = 30000
    n_test = 3461
    pca_k = houzz.Houzz.N_TEXT_FEATURE
    # Get the partition
    train_labels, test_labels = houzz.partition(n_test, n_train)
    results_dir = utilities.fullfile(houzz.DATASET_ROOT, 'results')

    with open(utilities.fullfile(results_dir, 'results_final.csv'), 'w') as log:
        # Write column names (text feature types)
        log.write(',')  # column for img feature type
        log.write(', '.join(TXT_FEATURE_TYPES))
        log.write('\n')

        for img_type in IMG_FEATURE_TYPES:
            log.write(img_type + ', ')
            for txt_type in TXT_FEATURE_TYPES:
                utilities.format_print(
                    'Compute Image {0} + Text {1}'.format(img_type, txt_type))

                # Skip null case
                if img_type is '' and txt_type is '':
                    log.write(', ')
                    continue

                # Train
                model, meta = learning.train(
                    img_feature_type=img_type,
                    txt_feature_type=txt_type,
                    feature_dir=houzz.DATASET_ROOT,
                    pca_k=pca_k,
                    n_test=n_test,
                    n_train=n_train,
                    train_labels=train_labels,
                    svm_dir=houzz.TRAINED_PATH)

                # Test
                confusion, acc = learning.test(model, meta, test_labels)

                # Record output
                utilities.format_print(
                    'Completed Image {0} + Text {1} with accuracy {2}'.format(
                        img_type, txt_type, acc))

                log.write(str(acc) + ', ')  # in cell img_type, txt_type
                model_path, _ = utilities.pack_name(
                    img_type, txt_type, pca_k, n_test, n_train, results_dir)

                fig_path = model_path[:-len('.model')] + '.png'
                utilities.plot(confusion, fig_path)
            # Go to next row in CSV
            log.write('\n')

if __name__ == '__main__':
    run()
