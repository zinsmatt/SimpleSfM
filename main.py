import glob
import logging
import random
import time

import numpy as np
import cv2

import features
import parallel
import dataset
import matching

logging.basicConfig(level=logging.DEBUG)

files = sorted(glob.glob("data/room/*.png"))

config = {
    "sift_peak_threshold": 0.1,
    "sift_edge_threshold": 10
}
features_count = 1000
num_proc = 16

t0 = time.time()

# M = 50

# images = []
# for f in files[:M]:
#     images.append(cv2.imread(f))


# arguments = []
# keypoints = []
# for i in range(10):
#     args = (images[i], config, features_count)
#     arguments.append(args)
#     # points, desc = features.extract_features_sift(images[i], config, features_count)
#     # keypoints.append((points, desc))
    
# keypoints = parallel.parallel_map(features.extract_features_sift_parallel, arguments, num_proc)

# print("done in {:.3f}s".format(time.time()-t0))
    
# # Display
# for i in random.sample(range(len(keypoints)), 3):
#     print("image", i)
#     image_kps = features.draw_features(images[i], keypoints[i][0], fixed_size=5)
#     cv2.imshow("features", image_kps)
#     cv2.waitKey()



dataset = dataset.DataSet("data/room")
res = matching.select_candidates_from_sequence([0, 100], list(range(200)), 10, 20)
print(res)