import numpy as np
import matplotlib.pyplot as plt
from imageio import imread, imwrite
from scipy.cluster.vq import vq,kmeans2

#[outputImg, meanColors] = quantizeRGB(origImg, k)
#return outputImg, meanColors
def  quantizeRGB(origImg,k):
    plt.imshow(origImg)
#    print(img.shape)
    row,col,d=origImg.shape
    num_cluster=k
    dataset=np.double(np.reshape(origImg,(row*col,d)))
    meanColors=np.zeros((k*d))
    #labels=np.zeros()
    meanColors,labels=kmeans2(dataset,num_cluster,iter=500,minit='random')
    outputImg=np.zeros((row,col,d),dtype='uint8')
    labels=np.reshape(labels,(row,col))
    for i in range(0,row):
        for j in range(0,col):
                #outputImg[i,j,d]=centroid[labels[i,j],:]
                outputImg[i,j,0]=meanColors[labels[i,j],0]
                #print(outputImg[i,j,0])
                outputImg[i,j,1]=meanColors[labels[i,j],1]
                outputImg[i,j,2]=meanColors[labels[i,j],2]
    plt.figure()
    return outputImg,meanColors


