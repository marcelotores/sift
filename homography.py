# importing the required libraries
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

imagem1 = 'images/motor1.jpg'
imagem2 = 'images/motor2.jpg'

# reading image in grayscale
img1 = cv.imread(imagem1, cv.IMREAD_GRAYSCALE)
img2 = cv.imread(imagem2, cv.IMREAD_GRAYSCALE)

MIN_MATCH_COUNT = 10

# creating the SIFT algorithm
sift = cv.SIFT_create()

# find the keypoints and descriptors with SIFT
kp_image1, desc_image1 = sift.detectAndCompute(img1, None)
kp_image2, desc_image2 = sift.detectAndCompute(img2, None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(desc_image1, desc_image2, k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp_image1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_image2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    #dst_pts = np.array([[318, 256], [534, 372], [316, 670], [73, 473]])
    print(src_pts)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

    im_out = cv.warpPerspective(img1, M, (img2.shape[1], img2.shape[0]))

    # Display images
    #cv.imshow("Source Image", img1)
    #cv.imshow("Destination Image", img2)
    cv.imshow("Imagem Fonte", cv.resize(img1, (660, 540)))
    cv.imshow("Imagem Destino", cv.resize(im_out, (660, 540)))
    cv.waitKey(0)

    cv.waitKey(0)
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None