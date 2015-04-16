import scipy.io


class Houzz:
    """A class for processing the Houzz.com dataset:
    Houzz
        - images
        - keys
        - metadata

    A single instance of this class represents the entire dataset.
    The 'loadmat()' method creates a Python dictionary
    for a single metadata file.
    """
    def __init__(self, base_folder, category):
        """ Assume Unix-style pathnames
        Remove trailing '/' from 'base_folder,' if present, to regularize input
        """
        base_folder = base_folder.rstrip('/')

        # Set instance variables
        self.base_folder = base_folder
        self.image_folder = base_folder + '/images/' + category
        self.keys_files = base_folder + '/keys/' + category + '.mat'
        self.data_folder = base_folder + '/metadata/' + category

    def loadmat(self, filename):
        """An example of the Matlab format of the Houzz dataset (data_02101038)
        data =
            url: [1x83 char]
             id: 2101038
     image_link: [1x74 char]
    description: [1x134 char]
          style: 'contemporary'
           type: 'bedroom'
            tag: {'built-in-desk'  'guest-room-retreat'  'vaulted-ceilings'}
       sameproj: {[1x77 char]  [1x83 char]  [1x87 char]}
      recommend: {1x6 cell}
        """
        py_mat = scipy.io.loadmat(filename, squeeze_me=True,
                                  struct_as_record=False)
        py_data = py_mat.data
        attributes = dir(py_data)

        data = dict()

        # for string array in matlab
        data['url'] = py_data.url if 'url' in attributes else None
        data['id'] = py_data.id if 'id' in attributes else None
        data['image_link'] = py_data.image_link \
            if 'image_link' in attributes else None
        data['description'] = py_data.description \
            if 'description' in attributes else None
        data['style'] = py_data.style if 'style' in attributes else None
        data['type'] = py_data.type if 'type' in attributes else None

        # for cell array in Matlab
        data['tag'] = [x for x in py_data.tag] if 'tag' in attributes else None
        data['sameproj'] = [x for x in py_data.sameproj] \
            if 'sameproj' in attributes else None
        data['recommend'] = [x for x in py_data.recommend] \
            if 'recommend' in attributes else None
        return data
