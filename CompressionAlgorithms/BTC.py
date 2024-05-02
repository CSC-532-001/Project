import numpy as np
from skimage.measure import block_reduce

# Block Truncation Coding

def btc_compress(img: np.ndarray, block=2):

    reduced_image = block_reduce(img, block, func=np.mean).astype(np.uint8)

    if len(img.shape) == 3:
        reduced_image = block_reduce(img, (block, block, 1), func=np.mean).astype(np.uint8)


    return reduced_image

