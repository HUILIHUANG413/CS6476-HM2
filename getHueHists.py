import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2hsv
#import quantizeHSV
from imageio import imread, imwrite
from scipy.cluster.vq import kmeans,vq
import copy
#[histEqual, histClustered, HSV] = getHueHists(im, k)
def getHueHist(im,k):
    row,col,d=im.shape
    #get the hsvImg
    hsvImg=np.double(rgb2hsv(im))
    hsvImg=hsvImg.reshape(row*col,3)
    #draw the hist of hsvImg
    flat_hsv=hsvImg[:,0]
    hue_range=(0.0,1.0)
    bins=np.zeros(k+1)
    for i in range(1,k+1):
        bins[i]=(1/k)*i
    [histimg_value,hist_bin] = np.histogram(flat_hsv,bins=bins,range=hue_range)
    #get the hist of quantizeHSV img
    quanti_hist=np.zeros(k) #build an array contain the number of pix in the same cluster
    meanHues,labels=kmeans(flat_hsv,k)
    distance=np.zeros(row*col)#compute the distance, find the closet meanHues
    for i in range(row*col):
        distance=(np.abs(meanHues-flat_hsv[i]))
        minIndex=np.argmin(distance)
        quanti_hist[minIndex]=quanti_hist[minIndex]+1
    return [[histimg_value,hist_bin],quanti_hist]


if __name__ == "__main__":
    img=imread('fish.jpg')
    k=10
    [histimg,histQuanti]=getHueHist(img,k)
    #computer the hue hist
    gap=(1/k)
    center=histimg[1][1:]-gap/2 #the center point
    plt.subplot(121)
    plt.bar(center,histimg[0],width=gap)
    #compute the second img
    centerQuanti=np.arange(k)+1
    plt.subplot(122)
    plt.bar(centerQuanti,histQuanti,width=1.0)
    plt.show()


