import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


#img = cv.imread('images/circulo.png')
img = cv.imread('images/44.jpeg')
gray= cv.cvtColor(img, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
#kp = sift.detect(gray, None)
kp, des = sift.detectAndCompute(gray,None)

img = cv.drawKeypoints(gray, kp, img)

cv.imwrite('sift_keypoints.jpg', img)

img = cv.drawKeypoints(gray, kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imwrite('sift_keypoints.jpg', img)

plt.imshow(img)    # show the train image keypoints
plt.title('Train Image Keypoints')
plt.show()

print(len(kp))

#print(des)

# Here kp will be a list of keypoints and des is a numpy array of shape (Number of Keypoints)×128.

# So we got keypoints, descriptors etc. Now we want to see how to match keypoints in different images. That we will learn in coming chapters.