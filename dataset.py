import os


class DataSet():
    """
        Accessors to the main input and output (images, masks, features,
        matches, ...)
    """
    def __init__(self, path):
        self.path = path

    def _images_list_file(self):
        return os.path.join(self.path, "images_list.txt")

    def load_images_list(self):
        with open(self._images_list_file(), "r") as fin:
            lines = fin.readlines()
        self.image_files = lines
        


    
