import sys
import cv2 as cv

imagem = sys.argv[1]
def get_descritores():

    img = cv.imread(imagem)
    gray2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    img2 = cv.imread('images/deformacao1.jpg')
    gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    for d in des:
        for d2 in des2:
            print(d2)

get_descritores()