import numpy as np
import matplotlib.pyplot as plt
from skimage import feature
from imageio import imread, imwrite
from skimage.color import rgb2gray, gray2rgb

# threshold you choose to put in the center
threshold=10
# min distance you choose between two center point
min_distance=10

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

# [centers] = detectCircles(im, radius, useGradient)
def detectCircles(im, radius, useGradient):
    row, col, d = im.shape
    # gray_im
    imGray = rgb2gray(im)
    # canny
    imEdge = feature.canny(imGray, sigma=3)
    # plt.subplot(131)
    imEdge = imEdge.astype(int) * 255
    # threshold
    # plt.figure()
    # plt.imshow(imEdge)
    # plt.show()

    # find theta
    theta_1 = np.zeros((row, col))
    # theta_1=compute_gradient(imEdge)
    theta_1_dx = np.zeros((row, col))
    theta_1_dy = np.zeros((row, col))
    theta_1_dx = np.gradient(imGray, axis=0)
    theta_1_dy = np.gradient(imGray, axis=1)
    theta_1 = np.arctan(-theta_1_dy / theta_1_dx)
    where_are_nan = np.isnan(theta_1)
    theta_1[where_are_nan] = np.pi / 2
    pace = np.int(radius * 3)
    ####### hough transform#######
    H = np.zeros((row, col))
    H_dict = dict()
    for i in range(0, row):
        for j in range(0, col):
            if imEdge[i, j] != 0:
                # use the gradient
                if useGradient == 1:

                    theta = theta_1[i, j]
                    a = np.int(i - radius * np.cos(theta))
                    b = np.int(j + radius * np.sin(theta))
                    # get larger acc matrix
                    for augmented_a_b in augment_a_b(a, b):
                        a_aug = augmented_a_b[0]
                        b_aug = augmented_a_b[1]
                        if (0 <= a_aug < row and 0 <= b_aug < col):
                            H[a_aug, b_aug] = H[a_aug, b_aug] + 1
                            H_dict[(a_aug, b_aug)] = H_dict.get((a_aug, b_aug), 0) + 1
                # do not use the gradient
                if useGradient == 0:
                    # theta=np.arange(0,2*np.pi,0.01*np.pi)
                    for angle in range(0, pace):
                        theta = 2 * np.pi * angle / pace
                        a = np.int(i - radius * np.cos(angle))
                        b = np.int(j + radius * np.sin(angle))
                        if (0 < a < row and 0 < b < col):
                            H[a, b] = H[a, b] + 1
                            H_dict[(a, b)] = H_dict.get((a, b), 0) + 1
    # center.append((0,0))
    plt.figure()
    plt.subplot(121)
    plt.title("accumulator arrays ")
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
    radius =8
    center = []
    center = detectCircles(im, radius,1)  # 45 for jupiter; 6 for eggs
    print(center)
    center = np.array(center)
    plt.subplot(122)
    # plt.imshow(im)
    x = np.zeros(center.shape[0])
    y = np.zeros(center.shape[0])
    x = center[:, 0]
    y = center[:, 1]
    plt.scatter(y, x, s=0.1, c='r')
    drawcircle(im, center, radius)
    #plt.suptitle('Circle_Detection_of_jupiter_r=100_useGradient=1')
    #plt.savefig("Circle_Detection_of_jupiter_r=100_useGradient=1.png")
    plt.show()






