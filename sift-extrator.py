import sys

import cv2
import numpy as np

imagem1 = sys.argv[1]

sift = cv2.SIFT_create()

image = cv2.imread(imagem1, cv2.IMREAD_GRAYSCALE)

# find keypoints in the image
kps = sift.detect(image)

# rank keypoints by importance (response value)
# and pick top 4 keypoints results

print(kps)
n = 4
kps = sorted(kps, key=lambda x: -x.response)[:n]

# compute descriptor values from keypoints (128 per keypoint)
kps, dsc = sift.compute(image, kps)

# vetor de característica com base na soma das 4 descritores
vector_4 = dsc[0][:10] + dsc[1][:10] + dsc[2][:10] + dsc[3][:10]

vector = dsc.flatten()

if vector.size < (n*128):
    # It can happen that there are simply not enough keypoints in an image,
    # in which case you can choose to fill the missing vector values with zeroes
    vector = np.concatenate([vector, np.zeros(n*128 - vector.size)])

print(vector[:10])
#print(vector)
# [ 0.  0.  0.  0.  0.  0.  0.  1. 21.  1.]