import numpy as np
from PIL import Image
from scipy.fftpack import *
from math import *

def loadImage(file):
    image = Image.open(file).convert('RGB') # Opens file into a bitmap and converts to RGB.
    # newLine = print()
    # print(image)
    imageArray = np.array(image)
    space2 = '  ' # Just used to make 2 spaces for formating
    print(space2 + 'Red' + space2 + 'Green' + space2 + 'Blue')
    print(imageArray)
    imageYCBCR = rgbToYCBCR(imageArray)
    print(imageYCBCR)
    return imageYCBCR

def rgbToYCBCR(image):
    # These numbers are a part of a formula, provided by ITU-R BT.601 standard, used in SDTV (Standard-Definition Television).
    nform = np.array([[0.299, 0.587, 0.114], # The Y row.
                     [-0.1687, -0.3313, 0.5], # The Cb row.
                     [0.5, -0.4187, -0.0813]]) # The Cr row.
    imageYCBCR = image.dot(nform.T) # Transposes nForm and uses dot multiplication.
    imageYCBCR[:,:,[1,2]] += 128 # Adds to Cb and Cr in every pixel, and normalizes chrominance channels.
    return np.uint8(imageYCBCR) # This will convert imageYCBCR to an unsigned 8-bit integer. It can clips values outside 0 - 255 range. This can cause minor lossyness.

def compressImage(image, blockSize = int(8), quality = int(50)):
    # height, width, _ = image.shape # Assigns variables to the tuple of image.shape from NumPy
    # compressedImage = np.zeros((height, width, 3), dtype = np.float32) # Creates a matrix 3 x 3 matrix full of zeros.

    # Pad the image
    originalHeight, originalWidth, _ = image.shape
    paddedImage = padImageToBlockSize(image, blockSize)
    height, width, _ = paddedImage.shape
    compressedImage = np.zeros((height, width, 3), dtype=np.float32)
    quantTable = quantizationTable(quality)
    
    print(compressedImage) # An example if you want to see it

    quantTable = quantizationTable(quality)
    print()
    print(quantTable)

    for channel in range(3):
        for i in range(0, height, blockSize):
            for j in range(0, width, blockSize):
                block = paddedImage[i:i+blockSize, j:j+blockSize, channel].astype(np.float32)
                if block.shape[0] == blockSize and block.shape[1] == blockSize:
                    dctBlock = blockDCT(block - 128)
                    quantizedBlock = np.round(dctBlock / quantTable)
                    idctBlock = blockIDCT(quantizedBlock * quantTable) + 128
                    compressedImage[i:i+blockSize, j:j+blockSize, channel] = idctBlock

    print(compressedImage)
    # Crop the padded areas off the compressed image
    compressedImage = compressedImage[:originalHeight, :originalWidth, :]

    return np.clip(compressedImage, 0, 255).astype(np.uint8)

    
def quantizationTable(quality):
    # Array is standardized by the JPEG(Joint Photographic Experts Group) image compression standard. https://www.sciencedirect.com/science/article/pii/S1742287608000285
    qTable = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])
    print(qTable)
    return qTable * (100 - quality) / 50

def blockDCT(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def blockIDCT(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def padImageToBlockSize(image, blockSize):
    height, width, _ = image.shape
    padHeight = (blockSize - height % blockSize) % blockSize
    padWidth = (blockSize - width % blockSize) % blockSize

    # Pad the bottom and right edges with the values from the edge pixels
    paddedImage = np.pad(image, 
                          ((0, padHeight), (0, padWidth), (0, 0)), 
                          'edge')
    return paddedImage
# print("Pick your quality 1-99, less quality means compression")
# uInput = int(input())

image = loadImage(r'Images\Soccer_Ball.jpg')
imageCompressed = compressImage(image)
result = Image.fromarray(imageCompressed, 'YCbCr').convert('RGB')
result.show()
result.save('Results/compressedImageSoccer.jpg', 'JPEG')
