import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from os.path import basename

import sys

imagem1 = sys.argv[1]
imagem2 = sys.argv[2]

nome_imagem = str(basename(imagem1))[:-4] + "-" + basename(imagem2)

img1 = cv.imread(imagem1, cv.IMREAD_GRAYSCALE)          # queryImage
img2 = cv.imread(imagem2, cv.IMREAD_GRAYSCALE)          # trainImage


#img = cv.imread('images/circulo.png')

#img = cv.imread('images/44.jpeg')
#gray= cv.cvtColor(img, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])


# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

print('Quantidade de Correspondências: ', len(good))


cv.imwrite(nome_imagem, img3)
print(nome_imagem)
fig = plt.figure()
ax = fig.add_subplot()
ax.text(2, 6, f'KeyPoints: {len(kp1)}:{len(kp2)}', fontsize=10)
ax.set_title(f'Correspondências: {len(good)}')

plt.imshow(img3), plt.show()

