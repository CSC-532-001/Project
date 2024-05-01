# Written by Landon Casstevens

# For running: py Algorithm_Analysis.py

import os
import time
import sewar.full_ref as metrics
import numpy as np

from DCT import rgbToYCBCR
from DWT import wavelet_transform
from BTC import btc_block
from pathlib import Path
from PIL import Image

# Getting rid of a useless sewar warning message. May need to disable during bug finding.
import warnings
warnings.filterwarnings("ignore")


def main():
    # Collects all images to be processed and their simple names (not filepaths).
    imagesToRun, names = directorySearch()
    print("Image\tAlgorithm\tRuntime\tMSE\tPSNR\tSSIM\tUQI")
    
    # # DCT Testing
    countDCT = 0
    for givenImage in (imagesToRun):
        dctAnalyzer(givenImage, names[countDCT])
        countDCT += 1

    # DWT Testing
    countDWT = 0
    for givenImage in (imagesToRun):
        dwtAnalyzer(givenImage, names[countDWT])
        countDWT += 1 

    # BTC Testing
    # countBTC = 0
    # for givenImage in (imagesToRun):
    #     btcAnalyzer(givenImage, names[countBTC])
    #     countBTC += 1 



def directorySearch():
    # This function returns all fullpaths and names for files in the Images folder for the project. 

    imageList = []
    namesList = []
    parentDirectory = Path(str(os.getcwd()))
    imageDirectory = str(parentDirectory) + str(r"\Images")
    for file in os.listdir(imageDirectory):
        filePath = str(os.path.abspath(imageDirectory) + "\\" + file)
        imageList.append(filePath)
        namesList.append(file)

    return imageList, namesList
        
    
def dctAnalyzer(givenImage, name):
    # Runs a given image through DCT. Prints all metrics calculated. 

    inputHolder = Image.open(givenImage)
    inputImage = np.array(inputHolder, dtype=np.ndarray)

    # DCT Analysis
    startTime = time.perf_counter()
    compressedImage = rgbToYCBCR(inputImage)
    endTime = time.perf_counter()
    # The four quality metrics used for this project. 
    MSE = metrics.mse(inputImage, compressedImage)
    PSNR = metrics.psnr(inputImage, compressedImage)
    SSIM = metrics.ssim(inputImage, compressedImage)
    UQI = metrics.uqi(inputImage, compressedImage)

    print(name + "\tDCT" + "\t" + (f"{endTime-startTime:.2f}") + "\t" + (f"{MSE:.2f}") + "\t" +  (f"{PSNR:.2f}") + "\t" +  str(SSIM) + "\t" +  (f"{UQI:.2f}"))

def dwtAnalyzer(givenImage, name):
    # Runs a given image through DWT. Prints all metrics calculated. 

    inputHolder = Image.open(givenImage)
    inputImage = np.array(inputHolder, dtype=np.uint8)

    # DWT Analysis
    startTime = time.perf_counter()
    compressedImage = wavelet_transform(inputImage)
    endTime = time.perf_counter()

    # Getting shape axes.
    shapeList = np.shape(inputImage)
    axisOne = shapeList[0]
    axisTwo = shapeList[1]

    # Image shape handling to display the four quality metrics used for this project.
    if len((np.shape(inputImage))) == 3:
        axisThree = shapeList[2]
        MSE = metrics.mse(inputImage, compressedImage[:axisOne,:axisTwo,:axisThree])
        PSNR = metrics.psnr(inputImage, compressedImage[:axisOne,:axisTwo,:axisThree])
        SSIM = metrics.ssim(inputImage, compressedImage[:axisOne,:axisTwo,:axisThree])
        UQI = metrics.uqi(inputImage, compressedImage[:axisOne,:axisTwo,:axisThree])
    # Image shape handling to display the four quality metrics used for this project. 
    if len((np.shape(inputImage))) == 2:
        MSE = metrics.mse(inputImage, compressedImage[:axisOne,:axisTwo])
        PSNR = metrics.psnr(inputImage, compressedImage[:axisOne,:axisTwo])
        SSIM = metrics.ssim(inputImage, compressedImage[:axisOne,:axisTwo])
        UQI = metrics.uqi(inputImage, compressedImage[:axisOne,:axisTwo])

    print(name + "\tDWT" + "\t" + (f"{endTime-startTime:.2f}") + "\t" + (f"{MSE:.2f}") + "\t" +  (f"{PSNR:.2f}") + "\t" +  str(SSIM) + "\t" +  (f"{UQI:.2f}"))

def btcAnalyzer(givenImage, name):
    # Runs a given image through BTC. Prints all metrics calculated. 

    inputHolder = Image.open(givenImage)
    inputImage = np.array(inputHolder, dtype=np.ndarray)

    # DCT Analysis
    startTime = time.perf_counter()
    compressedImage = btc_block(inputImage)
    endTime = time.perf_counter()
    # The four quality metrics used for this project. 
    MSE = metrics.mse(inputImage, compressedImage)
    PSNR = metrics.psnr(inputImage, compressedImage)
    SSIM = metrics.ssim(inputImage, compressedImage)
    UQI = metrics.uqi(inputImage, compressedImage)

    print(name + "\tBTC" + "\t" + (f"{endTime-startTime:.2f}") + "\t" + (f"{MSE:.2f}") + "\t" +  (f"{PSNR:.2f}") + "\t" +  str(SSIM) + "\t" +  (f"{UQI:.2f}"))

main()