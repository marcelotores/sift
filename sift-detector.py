import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import sys
from os.path import basename

imagem = sys.argv[1]
nome_imagem = basename(imagem)

img = cv.imread(imagem)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
kp, des = sift.detectAndCompute(gray, None)

# apenas o ponto chave
img = cv.drawKeypoints(gray, kp, img)

# círculo com tamanho de ponto-chave e sua orientação
img = cv.drawKeypoints(gray, kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv.imwrite(f'detector_{nome_imagem}', img)


fig = plt.figure()
ax = fig.add_subplot()
ax.text(2, 6, f'KeyPoints: {len(kp)}', fontsize=10)

plt.imshow(img)    # show the train image keypoints
plt.title(imagem)
plt.show()

