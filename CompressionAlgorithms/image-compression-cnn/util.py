import skimage.io
import skimage.transform
from skimage import img_as_ubyte
import numpy as np
from PIL import Image


def chunker(seq, size):
    # http://stackoverflow.com/a/25701576/1189865
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def load_image(path):
    try:
        img = skimage.io.imread(path).astype(float)
        img /= 255.0
        X = img.shape[0]
        Y = img.shape[1]
        S = min(X, Y)
        XX = int((X - S) / 2)
        YY = int((Y - S) / 2)

        # if black and white image, repeat the channels
        if len(img.shape) == 2:
            img = np.tile(img[:, :, None], 3)

        resized_img = skimage.transform.resize(
            img[XX : XX + S, YY : YY + S], [224, 224]
        )
        resized_img_uint8 = img_as_ubyte(resized_img)

        return resized_img_uint8

    except FileNotFoundError:
        return Exception("File not found: {}".format(path))
    except Exception as e:
        return e


def load_single_image(image):
    return np.expand_dims(load_image(image), 0)


# def load_image_tensorflow(path):
#     img = skimage.io.imread( path ).astype( float )
#     img_resized = tf.image.resize_image_with_crop_or_pad(tf.convert_to_tensor(img, dtype=tf.float32), 224, 224)
#     img_resized = tf.expand_dims(img_resized, 0)
#     return img_resized, img


def array2PIL(arr):
    mode = "RGBA"
    shape = arr.shape
    arr = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2])
    if len(arr[0]) == 3:
        arr = np.c_[arr, 255 * np.ones((len(arr), 1), np.uint8)]

    return Image.frombuffer(
        mode, (shape[1], shape[0]), arr.tostring(), "raw", mode, 0, 1
    )


def normalize(x):
    min = np.min(x)
    max = np.max(x)
    return (x - min) / (max - min)
