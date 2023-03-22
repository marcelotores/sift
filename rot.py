# import required libraries
import cv2 as cv
from os.path import basename, join
from os import listdir

def rotaciona_imagem(imagem):
    nome_imagem = basename(imagem)

    img = cv.imread(imagem)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    img_cw_90 = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
    img_ccw_90 = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
    img_cw_180 = cv.rotate(img, cv.ROTATE_180)

    cv.imwrite(f'images/imagens-deformadas/{nome_imagem[:-4]}90.jpg', img_cw_90)
    cv.imwrite(f'images/imagens-deformadas/{nome_imagem[:-4]}180.jpg', img_cw_180)
    cv.imwrite(f'images/imagens-deformadas/{nome_imagem[:-4]}270.jpg', img_ccw_90)

caminho = 'images/imagens-correspondencias'

for filename in listdir(caminho):
    if filename.endswith(''):
        with open(join(caminho, filename)) as f:
            imagem = caminho + '/' + filename
            rotaciona_imagem(imagem)
