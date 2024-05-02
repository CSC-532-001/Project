import numpy as np
import pywt


def wavelet_transform(img, ratio=0.5):
    compressed = compress(img, ratio)
    return decompress_image(compressed)


def compress(img, ratio):
    coeffs = pywt.wavedec2(img, "haar")

    flattened_coeffs = np.concatenate(
        [c.flatten() for level_coeffs in coeffs for c in level_coeffs]
    )
    sorted_indices = np.argsort(np.abs(flattened_coeffs))

    # Keep only the most important coefficients
    num_coefficients = int(len(sorted_indices) * ratio)
    kept_indices = sorted_indices[-num_coefficients:]
    flattened_coeffs[~np.isin(np.arange(len(flattened_coeffs)), kept_indices)] = 0

    # Reconstruct the compressed image
    compressed_coeffs = []
    idx = 0
    for level_coeffs in coeffs:
        level_compressed_coeffs = []
        for c in level_coeffs:
            if isinstance(c, tuple):
                for sub_c in c:
                    flat_len = np.prod(sub_c.shape)
                    level_compressed_coeffs.append(
                        flattened_coeffs[idx : idx + flat_len].reshape(sub_c.shape)
                    )
                    idx += flat_len
            else:
                flat_len = np.prod(c.shape)
                level_compressed_coeffs.append(
                    flattened_coeffs[idx : idx + flat_len].reshape(c.shape)
                )
                idx += flat_len
        compressed_coeffs.append(level_compressed_coeffs)

    return compressed_coeffs


def decompress_image(coeffs):
    return pywt.waverec2(coeffs, "haar")
