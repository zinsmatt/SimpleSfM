import glob
import logging
import random
import time

import numpy as np
import cv2

import features
import matching
import parallel
import dataset
import matching
import tracking

logging.basicConfig(level=logging.DEBUG)

files = sorted(glob.glob("data/room/*.png"))

config = {
    "sift_peak_threshold": 0.1,
    "sift_edge_threshold": 10
}
features_count = 800
num_proc = 16

t0 = time.time()

# M = 50

# images = []
# for f in files[:M]:
#     images.append(cv2.imread(f))

dataset = dataset.DataSet("data/room")

arguments = []
keypoints = []
for i in range(dataset.nb_images):
    args = (dataset.load_image(i), config, features_count)
    arguments.append(args)
    # points, desc = features.extract_features_sift(images[i], config, features_count)
    # keypoints.append((points, desc))
    
keypoints = parallel.parallel_map(features.extract_features_sift_parallel, arguments, num_proc)
for i in range(dataset.nb_images):
    dataset.set_features(i, keypoints[i])

# print("done in {:.3f}s".format(time.time()-t0))
    
# # Display
# for i in random.sample(range(len(keypoints)), 3):
#     print("image", i)
#     image_kps = features.draw_features(images[i], keypoints[i][0], fixed_size=5)
#     cv2.imshow("features", image_kps)
#     cv2.waitKey()


# res = matching.select_candidates_from_sequence([0, 100], list(range(200)), 10, 20)

# print(res)

matches = matching.match_images(dataset, {}, list(range(dataset.nb_images)), list(range(dataset.nb_images)))

# for (i, j), m in matches.items():
#     pts_i, _ = dataset.get_features(i)
#     pts_j, _ = dataset.get_features(j)
#     img_i = dataset.load_image(i)
#     img_j = dataset.load_image(j)

#     kps_i = [cv2.KeyPoint(x=p[0], y=p[1], _size=p[2]) for p in pts_i]
#     kps_j = [cv2.KeyPoint(x=p[0], y=p[1], _size=p[2]) for p in pts_j]

#     dmatches = [cv2.DMatch(_imgIdx=0, _queryIdx=a, _trainIdx=b, _distance=0) for a,b in m.tolist()]
#     img_match = cv2.drawMatches(img_i, kps_i, img_j, kps_j, dmatches, None, flags=2)
#     cv2.imshow("match", img_match)
#     cv2.waitKey()
tracks = tracking.create_tracks(dataset, matches)



images = {}
for track in tracks:
    line = []
    for im, f in track:
        if im not in images:
            images[im] = dataset.load_image(im)
        image = images[im]

        p = dataset.features[im][0][f, :2]
        line.append(p)

        for i in range(len(line)-1):
            cv2.line(image, (int(round(line[i][0])), int(round(line[i][1]))),
                     (int(round(line[i+1][0])), int(round(line[i+1][1]))), (0, 255, 255), 1)

        for p in line:
            cv2.circle(image, (int(round(p[0])), int(round(p[1]))), 2, (0, 255, 0), -1)

for im in sorted(images.keys()):
    cv2.imwrite("output/img_%04d.png" % im, images[im])
    cv2.imshow("tracks", images[im])
    cv2.waitKey()