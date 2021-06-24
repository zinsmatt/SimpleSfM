import bisect
from typing import Tuple, Dict, Any, List, Optional
import cv2

from dataset import DataSet

def match_images(
    dataset: DataSet, config: Dict[str, Any],
    ref_images: List[int], cand_images: List[int]
    ):
    return 0




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