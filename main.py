from learning import train_svm
from houzz import partition

(train, test) = partition()
train_svm('caffenet_text', train, '/home/chunwei/Data/houzz/caffenet_features')
