import cv2 as cv

#usar un descriptor de caracteristicas sift
image = cv.imread('img/tableroA.png')
image = cv.resize(image, (560, 560))
sift = cv.SIFT_create()

keypoints, descriptors = sift.detectAndCompute(image, None)

image = cv.drawKeypoints(image, keypoints, None)

cv.imshow('image', image)
cv.waitKey(0)
