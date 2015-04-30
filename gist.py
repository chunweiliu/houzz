import houzz
import numpy
import leargist
from format_print import format_print
from PIL import Image

def generate_features(txt_file, img_dir, output_dir):
    """
    Compute and pickle GIST descriptors
    """
    img_dir = houzz.standardize(img_dir)
    output_dir = houzz.standardize(output_dir)
    with open(txt_file, 'r') as dataset:
        for line in dataset:
            img_file = line.split()[0]
            # Load image and compute GIST
            im = Image.open(img_dir + img_file)
            gist = leargist.color_gist(im)

            # Save as .npy
            npy = img_file.replace('.jpg', '.npy')
            numpy.save(output_dir + npy, gist)
            format_print("Output written for {}".format(img_file))


