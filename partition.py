import houzz
import numpy as np

def partition(file_list='bedroom.txt'):
    """
    Partitions the dataset into training and test sets.

    @param file_list
        a text file in the format expected by Caffe.
        <xxxx.jpg> <int>
    """
    TEST_FRACTION = 10  # 1 / TEST_FRACTION for test set

    train = dict()
    test = dict()
    with open(file_list) as f:
        for i, line in enumerate(f):
            filename = line.split()[0].rstrip('.jpg')
            label = int(line.split()[1])
            if i % TEST_FRACTION == 0:
                test[filename] = label
            else:
                train[filename] = label
    return (train, test)
			
			
