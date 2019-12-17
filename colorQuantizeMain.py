import numpy as np
import matplotlib.pyplot as plt
from imageio import imread, imwrite
from scipy.cluster.vq import vq,kmeans2
import quantizeHSV
import quantizeRGB
import computeQuantizationError
import getHueHists

if __name__ == "__main__":
    img=imread('fish.jpg')
    k=3
    #the result of (a)
    RGB_quantize_img,meanColor=quantizeRGB.quantizeRGB(img,k)
    plt.subplot(221)
    plt.title("RGB  Quantization Image(k=3)")
    plt.imshow(RGB_quantize_img)
    #the result of (b)
    HSV_quantize_img,meanHue=quantizeHSV.quantizeHSV(img, k)
    plt.subplot(222)
    plt.title("HSV Quantization Image(k=3)")
    plt.imshow(HSV_quantize_img)
    #compute error
    RGB_error=computeQuantizationError.computeQuantizationError(img,RGB_quantize_img)
    HSV_error=computeQuantizationError.computeQuantizationError(img,HSV_quantize_img)
    print("The ssd of RGB:",RGB_error)
    print("The ssd of HSV:",HSV_error)
    #histogram
    [histimg, histQuanti] = getHueHists.getHueHist(img, k)
    # computer the hue hist
    gap = (1 / k)
    center = histimg[1][1:] - gap / 2  # the center point
    plt.subplot(223)
    plt.title("histogram of img(k=3)")
    plt.bar(center, histimg[0], width=gap)
    #plt.imshow()
    # compute the second img
    centerQuanti = np.arange(k) + 1
    plt.subplot(224)
    plt.bar(centerQuanti, histQuanti, width=1.0)
    plt.title("histogram of quantization img(k=3)")
    plt.suptitle("Result_for_quetion_1_k=3.png")
    plt.savefig("Result_for_quetion_1_k=3.png")
    plt.show()
