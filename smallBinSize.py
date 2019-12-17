import numpy as np
import matplotlib.pyplot as plt
from skimage import feature
from imageio import imread, imwrite
from skimage.color import rgb2gray, gray2rgb
from scipy.misc import imresize

# threshold you choose to put in the center
threshold=20
# min distance you choose between two center point
min_distance=10
#the w&h of the bin size, which means the image would be reduced by bin_size_factor times
bin_size_factor=3


# jupiter r=100, threeshold=38.min_distance=10 for usergradient=0, threshold=12 for usergradient=1
# jupiter r=51 ,threshold=20,min_distance=10  for usergraient=0;threshold=35 for usergradient=1
# egg r=8, threshold=10,min_distance=10
# egg r=5, threshold=9,min_distance=10 for usergraient=0, the threshold =12 for usergradient=1

def not_close_center(PointA, Keyset):
    for s in Keyset:
        if (PointA[0] - s[0]) ** 2 + (PointA[1] - s[1]) ** 2 <= min_distance ** 2:
            return False
    return True
def drawcircle(im, center, r):
    pace = 360
    for i in range(center.shape[0]):
        a = center[i, 0]
        b = center[i, 1]
        result_img = []
        for angle in range(0, 500):
            theta = 2 * np.pi * angle / pace
            x = np.int(a + r * np.cos(theta))
            y = np.int(b - r * np.sin(theta))
            if (0 <= x < row and 0 <= y < col):
                # im[x,y,0]=255
                # im[x,y,1]=0
                # im[x,y,2]=0
                result_img.append((y, x))
        result_img = np.array(result_img)
        x = np.zeros(result_img.shape[0])
        y = np.zeros(result_img.shape[1])
        x = result_img[:, 0]
        y = result_img[:, 1]
        plt.imshow(im)
        plt.scatter(x, y, s=0.1, c='r')
        # plt.Circle(xy=(b,a),radius=r);

    plt.title("Detect Image")
def augment_a_b(a, b):
    res = []
    augment = [[-1, -1], [-1, 0], [-1, 1],
               [0, -1], [0, 0], [0, 1],
               [1, -1], [1, 0], [1, 1]]
    for aug in augment:
        res.append((a + aug[0], b + aug[1]))
    return res

def detectCircles(im, radius):
    row, col, d = im.shape
    # gray_im
    imGray = rgb2gray(im)
    # canny
    imEdge = feature.canny(imGray, sigma=3)
    plt.figure()
    plt.imshow(imEdge)
    pace = np.int(radius * 3)
    ####### hough transform#######
    H = np.zeros((row// bin_size_factor, col//bin_size_factor))
    H_dict = dict()
    for i in range(0, row):
        for j in range(0, col):
            if imEdge[i, j] != 0:
                # do not use the gradient
                # theta=np.arange(0,2*np.pi,0.01*np.pi)
                for angle in range(0, pace):
                    theta = 2 * np.pi * angle / pace
                    a = int((-radius * np.cos(theta) + i) / bin_size_factor)
                    b = int((radius * np.sin(theta) + j) / bin_size_factor)
                    if (0 <= a <(row//bin_size_factor) and 0 <=b <(col//bin_size_factor)):
                        H[a, b] = H[a, b] + 1
                        print(H[a,b])
                        H_dict[(a, b)] = H_dict.get((a, b), 0) + 1
    plt.figure()
    plt.figure()
    plt.subplot(121)
    plt.title("Accumulator_arrays_3times_less ")
    plt.imshow(H)
    # find the max center
    H_sorted = sorted(H_dict.items(), key=lambda kv: kv[1], reverse=True)
    center = []
    for key, value in H_sorted:
        if value < threshold:
            break
        else:
            if not_close_center(key, center):
                center.append(key)
    return center

if __name__ == "__main__":
    im = imread('egg.jpg')
    # im=gray2rgb(im)
    plt.imshow(im)
    row, col, d = im.shape
    center = []
    radius = 5
    center = detectCircles(im, radius)  # 45 for jupiter; 6 for eggs
    print(center)
    center = np.array(center)
    plt.subplot(122)
    # plt.imshow(im)
    x = np.zeros(center.shape[0])
    y = np.zeros(center.shape[0])
    x = center[:, 0]
    y = center[:, 1]
    plt.scatter(y, x, s=0.1, c='r')
    im=imresize(im,size=(row// bin_size_factor, col//bin_size_factor))
    drawcircle(im, center, radius)
    plt.suptitle('Circle_Detection_of_egg_r=4_lessbin.png')
    plt.savefig("Circle_Detection_of_egg_r=4_lessbin.png")
    plt.show()