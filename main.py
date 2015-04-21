# import svmutil
import liblinearutil

from learning import train_svm
from learning import confusion
from learning import load_dataset
# from learning import debug
from houzz import partition

# Training
model_name = 'caffenet_text_small'
img_feature_dir = '/home/chunwei/Data/houzz/caffenet_features'
(train_labels, test_labels) = partition(test_fraction=10, stop=500)
train_svm(model_name, train_labels, img_feature_dir)

# Testing
model_file = '/home/chunwei/Data/houzz/trained_svms/' + model_name + '.model'
# model = svmutil.svm_load_model(model_file)
model = liblinearutil.load_model(model_file)

x, y = load_dataset(test_labels, img_feature_dir)
# x, y = debug(x, y)
# print "DEBUG"
# print y.count(0)
# print y.count(1)
# predicted_labels, stats, p_val = svmutil.svm_predict(y, x, model, '-q')
predicted_labels, stats, p_val = liblinearutil.predict(y, x, model, '-q')

accuracy = stats[0]

# Visualize the result
mat = confusion(y, predicted_labels)
print mat
print accuracy

# print predicted_labels
# print stats
