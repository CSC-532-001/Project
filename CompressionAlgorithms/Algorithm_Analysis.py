# Written by Landon Casstevens

# For running: py Algorithm_Analysis.py

import os
import time
import sewar.full_ref as metrics
import numpy as np

from DCT import rgbToYCBCR
from pathlib import Path
from PIL import Image

# Getting rid of a useless sewar warning message. May need to disable during bug finding.
import warnings
warnings.filterwarnings("ignore")


def main():
    
    imagesToRun, names = directorySearch()
    print("Image\tAlgorithm\tRuntime\tMSE\tPSNR\tSSIM\tUQI")
    count = 0
    for givenImage in (imagesToRun):
        dctAnalyzer(givenImage, names[count])
        count += 1

def directorySearch():

    # This function will run 

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
    # Runs a given image through DCT, DWT, and JPG. Prints all metrics calculated. 

    inputHolder = Image.open(givenImage)
    inputImage = np.array(inputHolder, dtype=np.uint8)

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

main()