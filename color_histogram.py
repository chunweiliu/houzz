"""
Compute HSV color histograms for each image in the dataset.
Omits grayscale images.

Brian Cristante
"""
from colorsys import rgb_to_hsv
import numpy
from scipy.misc import imread

import houzz
from format_print import format_print

# Standard RGB max values (used by scipy.misc.imread())
STD_MAX_R = 255.0
STD_MAX_G = 255.0
STD_MAX_B = 255.0

# HSV max values, according to colorsys
MAX_H, MAX_S, MAX_V = 1.0, 1.0, 1.0

"""
HSVHistogram
Data structure for HSV histograms

Methods:
    - bin(self, h, s, v):
        add HSV value to the histogram
    - count(self, h_idx, s_idx, v_idx):
        get frequency from a bin
    - as_list()
        return the histogram contents as a vector
"""


class HSVHistogram(object):
    # Class constants
    NUM_BINS = 10
    H_BIN_SIZE = MAX_H/NUM_BINS
    S_BIN_SIZE = MAX_S/NUM_BINS
    V_BIN_SIZE = MAX_V/NUM_BINS

    def __init__(self):
        self.total = 0  # total number of entries; used to normaliize
        # Creates a 3-D double array w/ each entry init to 0.0
        size = (self.NUM_BINS, self.NUM_BINS, self.NUM_BINS)
        self.hist = numpy.zeros(size)

    def bin(self, h, s, v):
        """
        @param HSV value
        Side Effect: bins value and updates histogram
        """
        # Bin number = floor(val/BIN_SIZE)
        # Max value -> last bin (formula causes out of bounds)
        h_idx = self.NUM_BINS-1 if approx_equals(h, MAX_H) else int(h/self.H_BIN_SIZE) 
        s_idx = self.NUM_BINS-1 if approx_equals(s, MAX_S) else int(s/self.S_BIN_SIZE)
        v_idx = self.NUM_BINS-1 if approx_equals(v, MAX_V) else int(v/self.V_BIN_SIZE)
         
        # Add to bin count
        self.hist[h_idx, s_idx, v_idx] += 1
        self.total += 1

    def count(self, h_idx, s_idx, v_idx):
        """
        @param bin coordinate in {0, 1, ..., NUM_BINS}
        @return normalized count
        """
        if self.total == 0:
            return 0
        else:
            return self.hist[h_idx, s_idx, v_idx]/self.total
            # hist is an ndarray of floats

    def as_list(self):
        """
        @return (list): histogram of color frequencies
        """
        to_return = []
        for i in xrange(0, self.NUM_BINS):
            for j in xrange(0, self.NUM_BINS):
                for k in xrange(0, self.NUM_BINS):
                    to_return.append(self.count(i, j, k))

        return to_return


def to_rgb(im):
    # This should be fsater than 1, as we only
    # truncate to uint8 once (?)
    w, h = im.shape
    ret = numpy.empty((w, h, 3), dtype=numpy.uint8)
    ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = im
    return ret


def hsv_hist(im_filename):
    """
    Compute a color histogram representation of an image.

    @param im_filename (str): image filename
    @return (HSVHistogram): 10-bin HSV color histogram
    @return None if given a grayscale image
    """
    I = imread(im_filename)   # image as a numpy ndarray
    # Check grayscale
    if len(I.shape) != 3:
        I = to_rgb(I)

    hist = HSVHistogram()     # create a new histogram

    # Bin each pixel in the image
    for row in I:
        for pixel in row:
            r, g, b = pixel
            # Scale to range [0.0, 1.0] expected by colorsys
            r /= STD_MAX_R
            g /= STD_MAX_G
            b /= STD_MAX_B
            h, s, v = rgb_to_hsv(r, g, b)

            hist.bin(h, s, v)

    print("Processed " + im_filename)
    return hist


def approx_equals(x, y):
    """
    Check floating-point "equality," up to some tolerance.

    @param two floating point values, x and y
    @return True or False
    """
    tol = 10**(-5)
    return True if abs(x - y) < tol else False


def generate_features(txt_file, img_dir, output_dir):
    """
    Compute and pickle HSV color histograms for the images
    listed in txt_file.
    """
    img_dir = houzz.standardize(img_dir)
    output_dir = houzz.standardize(output_dir)
    with open(txt_file, 'r') as dataset:
        for line in dataset:
            img_file = line.split()[0]
            hist = hsv_hist(img_dir + img_file)
            # transform hist to array
            hist = numpy.array(hist.as_list())
            npy = img_file.replace('.jpg', '.npy')
            numpy.save(output_dir + npy, hist)

            format_print("Output written for {}".format(img_file))

