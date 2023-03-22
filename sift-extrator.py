import sys

import cv2
import numpy as np
from os.path import join
from os import listdir

def extrator_de_caracteristicas(imagem1):

    cat1 = 'motor'
    cat2 = 'sensor'
    cat3 = 'bal'

    sift = cv2.SIFT_create()

    image = cv2.imread(imagem1, cv2.IMREAD_GRAYSCALE)

    # find keypoints in the image
    kps = sift.detect(image)

    # rank keypoints by importance (response value)
    # and pick top 4 keypoints results
    n = 4
    kps = sorted(kps, key=lambda x: -x.response)[:n]

    # compute descriptor values from keypoints (128 per keypoint)
    kps, dsc = sift.compute(image, kps)

    # vetor de característica com base na soma das 4 descritores
    # vector_4 = dsc[0][:10] + dsc[1][:10] + dsc[2][:10] + dsc[3][:10]
    vector = dsc.flatten()

    if vector.size < (n * 128):
        # It can happen that there are simply not enough keypoints in an image,
        # in which case you can choose to fill the missing vector values with zeroes
        vector = np.concatenate([vector, np.zeros(n * 128 - vector.size)])


    if cat1 in imagem1:
        cat = '1:'
    elif cat2 in imagem1:
        cat = '2:'
    elif cat3 in imagem1:
        cat = '3:'
    else:
        cat = False

    with open("features-4.txt", "a") as file:
        # Writing data to a file
        file.write(cat)
        file.write(str(vector)+"\n\n")


    #return vector

caminho = 'images/imagens-correspondencias'

for filename in listdir(caminho):
    if filename.endswith(''):
        with open(join(caminho, filename)) as f:
            imagem = caminho + '/' + filename
            extrator_de_caracteristicas(imagem)

#extrator_de_caracteristicas('images/imagens-correspondencias/sensor1.jpg')

# with open("features-4.txt", "w") as file:
#     # Writing data to a file
#     file.write(str(vector))
#     #file1.writelines(L)
#print(vector)
# [ 0.  0.  0.  0.  0.  0.  0.  1. 21.  1.]