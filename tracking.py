import logging
from unionfind import UnionFind

logger = logging.getLogger(__name__)


def create_tracks(dataset, matches):
    logger.debug("Merging features into tracks")
    uf = UnionFind()
    for im1, im2 in matches:
        for f1, f2 in matches[im1, im2]:
            uf.union((im1, f1), (im2, f2))
    # create a dict -> {root: [list of connected nodes]}
    sets = {}
    for i in uf:
        p = uf[i]
        if p in sets:
            sets[p].append(i)
        else:
            sets[p] = [i]

    min_length = 3
    good_tracks = [t for t in sets.values() if _good_track(t, min_length)]
    logger.debug("Good tracks: {}".format(len(good_tracks)))
    return good_tracks


def _good_track(track, min_length):
    if len(track) < min_length:
        return False
    images = [t[0] for t in track]
    if len(images) != len(set(images)):
        return False
    return True