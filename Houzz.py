import scipy.io


class Houzz:
    """A class for processing the Houzz.com dataset:
    Houzz
    - images
    - keys
    - metadata
    """
    def __init__(self, base_folder, category):
        self.base_folder = base_folder

        self.image_folder = base_folder + 'images/' + category
        self.keys_files = base_folder + 'keys/' + category + '.mat'
        self.data_folder = base_folder + 'metadata/' + category

    def loadmat(self, filename):
        """An example of the Matlab format of the Houzz dataset
        data =
               url: [1x113 char]
                id: 4945307
        image_link: [1x75 char]
             style: 'mediterranean'
              type: 'bedroom'
               tag: {''Bedroom-en-suite-%2F-Dar-Farah'}
          sameproj: {1x15 cell}
         recommend: {1x6 cell}
        """
        py_mat = scipy.io.loadmat(filename)
        data = dict()
        data['url'] = py_mat['data']['url'][0][0][0]
        data['id'] = py_mat['data']['id'][0][0][0][0]
        data['image_link'] = py_mat['data']['image_link'][0][0][0]
        data['style'] = py_mat['data']['style'][0][0][0]
        data['type'] = py_mat['data']['type'][0][0][0]
        data['tag'] = py_mat['data']['tag'][0][0][0][0][0]
        data['sameproj'] = py_mat['data']['sameproj'][0][0][0]  # 1 x 15 array
        data['recommend'] = py_mat['data']['recommend'][0][0][0]  # 1 x 6 array
        return data
