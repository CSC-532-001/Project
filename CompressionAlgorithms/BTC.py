import numpy as np
from skimage.measure import block_reduce

def btc_compress(img: np.ndarray, block=8):
    if len(img.shape) == 2: 
        reduced_image = block_reduce(img, (block, block), func=np.mean)
        expanded = np.repeat(np.repeat(reduced_image, block, axis=0), block, axis=1)
        expanded = expanded[:img.shape[0], :img.shape[1]]
    else:
        reduced_image = block_reduce(img, (block, block, block), func=np.mean)
        expanded = np.repeat(np.repeat(reduced_image, block, axis=0), block, axis=1)
        expanded = expanded[:img.shape[0], :img.shape[1], :img.shape[2]]

    return np.where(img > expanded, 1, 0)
