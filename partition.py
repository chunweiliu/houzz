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
                test[filename] = (feature(filename), label)
            else:
                train[filename] = (feature(filename), label)
    return (train, test)


def feature(filename, img_dir, txt_dir=houzz.DATASET_ROOT + 'text_features'):
	"""
	Preconditions:
		1) Image features precomputed and stored in img_dir
		2) Text features precomputed and stored in txt_dir
	@param
		filename (str): data_xxxx
		img_dir (str): location of image features 
		txt_dir (str): location of text features 

	@returns
		Combined img + text feature representation
	"""
	img_dir = standardize(img_dir)
	txt_dir = standardize(txt_dir)

	# Load features from .npy files
	img = np.load(img_dir + filename + '.npy')
	txt = np.load(txt_dir + filename + '.npy')
	
	# Concatenate
	return np.concatenate( (img.flatten(), txt.flatten()) )
			
			
