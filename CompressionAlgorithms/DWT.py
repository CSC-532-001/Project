import numpy as np
import pywt


def wavelet_transform(img, ratio=0.5):
    compressed = compress(img, ratio)
    return decompress_image(compressed)


def compress(img, ratio):
    coeffs = pywt.wavedec2(img, "haar", 2)

    flattened_coeffs = np.concatenate([c.flatten() for c in coeffs])
    sorted_indices = np.argsort(np.abs(flattened_coeffs))

    # keep only the most important coefficients
    num_coefficients = int(len(sorted_indices) * ratio)
    kept_indices = sorted_indices[-num_coefficients:]
    flattened_coeffs[~np.isin(np.arange(len(flattened_coeffs)), kept_indices)] = 0

    # Reconstruct the compressed image
    compressed_coeffs = []
    idx = 0
    for c in coeffs:
        flat_len = len(c.flatten())
        compressed_coeffs.append(
            flattened_coeffs[idx : idx + flat_len].reshape(c.shape)
        )
        idx += flat_len

    return compressed_coeffs


def decompress_image(coeffs):
    return pywt.waverec2(coeffs, "haar")
