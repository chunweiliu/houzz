import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2


def image_feature(text_file, lmdb_folder, output_folder):
    """Parse the file names and labels in the text_file, search for the
    corresponding features in the lmdb database, write an individual file
    for each record in the output_folder
    """
    lmdb_env = lmdb.open(lmdb_folder)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe_pb2.Datum()

    with open(text_file, 'r') as f:

        for key, value in lmdb_cursor:
            datum.ParseFromString(value)

            # label = datum.label
            data = caffe.io.datum_to_array(datum)
            filename = f.readline().split()[0]
            output_file = output_folder + filename.rstrip('.jpg') + '.npy'
            np.save(output_file, data)
