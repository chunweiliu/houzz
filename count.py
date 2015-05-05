import collections

import houzz

counts = collections.defaultdict(float)
_, test_labels = houzz.partition(n_test=3461, n_train=30000)
for label in test_labels.values():
        counts[label] += 1.0

print "chance: {}".format(counts.values()[0]/sum(counts.values()))
