import numpy as np
import matplotlib.pyplot as plt
from imageio import imread, imwrite
from scipy.cluster.vq import vq,kmeans2
from skimage.color import rgb2hsv,hsv2rgb
import matplotlib.colors as matColor
import copy
#[outputImg, meanHues] = quantizeHSV(origImg, k)

def quantizeHSV(origImg,k):

    row,col,d=origImg.shape
    num_cluster=k
    HSVimg=matColor.rgb_to_hsv(origImg)
    dataset=np.reshape(HSVimg,(row*col,d))
    hueImg=np.double(dataset[:,0])
    meanHues, labels = kmeans2(hueImg, k)
    outputImg = HSVimg.copy()
    for i in range(row):
        for j in range(col):
            #print("i=",i,"j=",j)
            distance=np.abs(meanHues-outputImg[i][j][0])
            min=np.argmin(distance)
            outputImg[i,j,0]=meanHues[min]
    meanHues = meanHues.flatten()
    outputImg = matColor.hsv_to_rgb(outputImg)
    return outputImg,meanHues


if __name__ == "__main__":
    img=imread('fish.jpg')
    #plt.imshow(img)
    k=8
    outIMG,mean=quantizeHSV(img,k)
    plt.imshow(outIMG)
    plt.show()