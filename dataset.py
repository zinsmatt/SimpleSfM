import logging
import os
import time
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class DataSet():
    """
        Accessors to the main input and output (images, masks, features,
        matches, ...)
    """

    def __init__(self, path):
        self.path = path
        self.image_files = []
        self.features = {}
        self.cameras = []
        self.load_images_list()
        self.load_cameras()

    @property
    def nb_images(self):
        return len(self.image_files)

    def _images_list_file(self):
        return os.path.join(self.path, "image_list.txt")

    def load_images_list(self):
        with open(self._images_list_file(), "r") as fin:
            lines = fin.readlines()
        self.image_files = [l.strip() for l in lines[:50:5]]

    def load_image(self, idx):
        if idx >= 0 and idx < self.nb_images:
            img = cv2.imread(self.image_files[idx])
            return cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
        else:
            logger.warning("Wrong image index")
            return None

    def load_cameras(self):
        K = np.array([[500.0, 0.0, 960.0],
                      [0.0, 500.0, 540.0],
                      [0.0, 0.0, 1.0]])
        for i in range(self.nb_images):
            self.cameras.append({"K": K})
    
    def get_camera(self, idx):
        if idx >= 0 and idx < self.nb_images:
            return self.cameras[idx]
        else:
            logger.warning("Wrong camera index")
            return None

    def set_features(self, idx, features):
        self.features[idx] = features

    def get_features(self, idx):
        if idx in self.features.keys():
            return self.features[idx]
        else:
            logger.warning("Wrong image index")
            return None


    def build_flann_index(self):
        # FLANN_INDEX_LINEAR = 0
        FLANN_INDEX_KDTREE = 1
        FLANN_INDEX_KMEANS = 2
        # FLANN_INDEX_COMPOSITE = 3
        # FLANN_INDEX_KDTREE_SINGLE = 4
        # FLANN_INDEX_HIERARCHICAL = 5
        FLANN_INDEX_LSH = 6

        t0 = time.time()
        self.flann_index = {}
        for i in range(self.nb_images):
            if i in self.features.keys():
                if self.features[i][1] is not None:
                    index = cv2.flann_Index(self.features[i][1], {
                                            'algorithm': FLANN_INDEX_KDTREE,
                                            'trees': 4
                    })
                    self.flann_index[i] = index
                else:
                    logger.debug("invalid features for index {}".format(i))

        logger.debug("Built {} flann index in {:.2f}s".format(
            len(self.flann_index),
            time.time()-t0
        ))


    def get_flann_index(self, idx):
        if idx in self.flann_index.keys():
            return self.flann_index[idx]
        else:
            logger.warning("No flann index")
            return None




        


