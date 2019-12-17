
import matplotlib.pyplot as plt
import quantizeRGB
from imageio import imread, imwrite
import numpy as np
import copy
from scipy.cluster.vq import vq,kmeans2
import quantizeHSV

#function [error] = computeQuantizationError(origImg,quantizedImg)
def computeQuantizationError(origImg,quantizedImg):
    row,col,_=origImg.shape
    SSD= 0.0
    for i in range(row):
        for j in range(col):
            flat_ori= origImg[i][j].flatten().astype(np.int)
            flat_quanti= quantizedImg[i][j].flatten().astype(np.int)
            error = np.linalg.norm(flat_ori - flat_quanti)
            SSD=error**2 +SSD
    return SSD.astype(np.int64)

if __name__ == "__main__":
    img = imread('fish.jpg')
    k = 3
    n = np.array([3,8,15])
    plt.imshow(img)
    for k in n:
        quantizedImg = img.copy
        quantizedImg, meancolor = quantizeHSV.quantizeHSV(img, k)
        error = computeQuantizationError(img, quantizedImg)
        print("k=", k, "error=", error)


