import numpy as np
from PIL import Image
from scipy.fftpack import *
from math import *

def loadImage(file):
    image = Image.open(file).convert('RGB') # Opens file into a bitmap and converts to RGB.
    # newLine = print()
    # print(image)
    imageArray = np.array(image)
    space2 = '  ' # Just used to make 2 spaces for formatiing
    print(space2 + 'Red' + space2 + 'Green' + space2 + 'Blue')
    print(imageArray)
    imageYCBCR = rgbToYCBCR(imageArray)
    print(imageYCBCR)

def rgbToYCBCR(image):
    # These numbers are a part of a formula, provided by ITU-R BT.601 standard, used in SDTV (Standard-Definition Television).
    nform = np.array([[0.299, 0.587, 0.114], # The Y row.
                     [-0.1687, -0.3313, 0.5], # The Cb row.
                     [0.5, -0.4187, -0.0813]]) # The Cr row.
    imageYCBCR = image.dot(nform.T) # Transposes nForm and uses dot multiplication.
    imageYCBCR[:,:,[1,2]] += 128 # Adds to Cb and Cr in every pixel, and normalizes chrominance channels.
    return np.uint8(imageYCBCR) # This will convert imageYCBCR to an unsigned 8-bit integer. It can clips values outside 0 - 255 range. This can cause minor lossyness.

loadImage(r'Images\Soccer_Ball.jpg')
