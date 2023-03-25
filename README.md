# sift

## Funções Importantes

sift.detect() 
-> Function finds the keypoint in the images. You can pass a mask if you want to search only a part of image
-> Each keypoint is a special structure which has many attributes like its (x,y) coordinates, size of the meaningful neighbourhood,
   angle which specifies its orientation, response that specifies strength of keypoints etc.
   
 ```
 kp = sift.detect(imagem, None)
 ## None seria a máscara?
 ```
   
cv.drawKeyPoints()
-> function which draws the small circles on the locations of keypoints.
-> cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS: flag para desenhar pontos chaves como mais detalhes.

```
## sem a flag
img=cv.drawKeypoints(gray,kp,img)

## com a flag
img=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

```

sift.compute()
-> computes the descriptors from the keypoints we have found

```
kp,des = sift.compute(imagem,kp)
```

sift.detectAndCompute()
-> find keypoints and descriptors in a single step
```
kp, des = sift.detectAndCompute(imagem,None)
# None seria a máscara
```


