import bisect
from typing import Tuple, Dict, Any, List, Optional
import cv2

from dataset import DataSet

def match_images(
    dataset: DataSet, config: Dict[str, Any],
    ref_images: List[int], cand_images: List[int]
    ):
    dataset.build_flann_index()
    selections = select_candidates_from_sequence(
        ref_images, cand_images, 5, 10
    )

    pairs = []
    for k, indices in selections.items():
        for i in indices:
            pairs.append((k, i))

    return match_images_pairs(dataset, pairs)


def match_images_pairs(dataset: DataSet, pairs: List[Tuple]):
    """ Perform pair matching. """
    matches = {}
    for im1, im2 in pairs:
        kps1, desc1 = dataset.get_features(im1)
        kps2, desc2 = dataset.get_features(im2)
        cam1 = dataset.get_camera(im1)
        cam2 = dataset.get_camera(im2)
        flann1 = dataset.get_flann_index
        m = match_flann(dataset, im1, im2)
        matches[im1, im2] = m
    return matches


def match_flann(dataset: DataSet, im1, im2):
    flann1 = dataset.get_flann_index(im1)
    flann2 = dataset.get_flann_index(im2)
    kps1, desc1 = dataset.get_features(im1)
    kps2, desc2 = dataset.get_features(im2)
    cam1 = dataset.get_camera(im1)
    cam2 = dataset.get_camera(im2)

    # search_params = dict(checks=config["flann_checks"])
    search_params = {}
    results, dists = flann1.knnSearch(desc2, 2, params=search_params)
    # results: indices of the k-nn of each feature in desc2
    # dists: distances associated
    squared_ratio = 0.8 ** 2  # Flann returns squared L2 distances

    # squared_ratio = config["lowes_ratio"] ** 2  # Flann returns squared L2 distances
    good = dists[:, 0] < squared_ratio * dists[:, 1]

    return list(zip(results[good, 0], good.nonzero()[0])) # returns a list of pairs (desc_1_i, desc_2_j)


def find_ge(a, x):
    """ Find leftmost item greater than or equal to x """
    i = bisect.bisect_left(a, x)
    if i != len(a):
        return i, a[i]
    return -1, None

def find_le(a, x):
    """ Find rightmost value less than or equal to x """
    i = bisect.bisect_right(a, x)
    if i:
        return i-1, a[i-1]
    return -1, None


def select_candidates_from_sequence(ref_images, cand_images, min_interval, max_interval):
    sorted_cand = sorted(cand_images)
    selections = {}
    for i in ref_images:
        selection = []
        left, _ = find_ge(sorted_cand, i - max_interval)
        right, _ = find_le(sorted_cand, i - min_interval)
        if left != -1 and right != -1 and left <= right:
            selection.extend(sorted_cand[left:right+1])

        left, _ = find_ge(sorted_cand, i + min_interval)
        right, _ = find_le(sorted_cand, i + max_interval)
        if left != -1 and right != -1 and left <= right:
            selection.extend(sorted_cand[left:right+1])
        selections[i] = selection

    return selections