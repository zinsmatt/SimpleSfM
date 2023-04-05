import logging
import time
from typing import Tuple, Dict, Any, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def extract_features_sift(
    image: np.ndarray, config: Dict[str, Any], features_count: int
) -> Tuple[np.ndarray, np.ndarray]:
    sift_edge_threshold = config["sift_edge_threshold"]
    sift_peak_threshold = float(config["sift_peak_threshold"])
    t = time.time()

    descriptor = cv2.SIFT_create(edgeThreshold=sift_edge_threshold, contrastThreshold=sift_peak_threshold)

    while True:
        detector = cv2.SIFT_create(edgeThreshold=sift_edge_threshold, contrastThreshold=sift_peak_threshold)
        points = detector.detect(image)

        logger.debug("Found {0} points in {1}s".format(len(points), time.time() - t))

        if len(points) < features_count and sift_peak_threshold > 0.0001:
            logger.debug("not enough points, reducing the threshold")
            sift_peak_threshold *= 0.75
        else:
            logger.debug("done")
            break

    points, desc = descriptor.compute(image, points)
    points = np.array([[p.pt[0], p.pt[1], p.size, p.angle] for p in points])

    return points, desc

def extract_features_sift_parallel(
    args: Tuple[np.ndarray, Dict[str, Any], int]
) -> Tuple[np.ndarray, np.ndarray]:
    return extract_features_sift(*args)


def draw_features(
    image: np.ndarray,
    points: np.ndarray,
    fixed_size: int=0
) -> np.ndarray:
    for p in points:
        s = fixed_size if fixed_size > 0 else int(round(p[2]))
        cv2.circle(image, (round(int(p[0])), round(int(p[1]))), s, (0, 255, 0))
    return image

