import cv2
from joblib import Parallel, delayed, parallel_backend


def parallel_map(func, args, num_proc, max_batch_size=1):
    # Disable OpenCV internal multi-threading
    threads_used = cv2.getNumThreads()
    cv2.setNumThreads(0)

    num_proc = min(num_proc, len(args))
    if num_proc <= 1:
        res = list(map(func, args))
    else:
        with parallel_backend("threading", n_jobs=num_proc):
            batch_size = max(1, len(args) // num_proc)
            if max_batch_size:
                batch_size = min(batch_size, max_batch_size)
            res = Parallel(batch_size=batch_size)(delayed(func)(arg) for arg in args)

    # Restore OpenCV internal multi-threading
    cv2.setNumThreads(threads_used)
    return res