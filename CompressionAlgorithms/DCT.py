from scipy.fftpack import dct


def rgbToYCBCR(img):
    return dct(dct(img, axis=0, norm="ortho"), axis=1, norm="ortho")
