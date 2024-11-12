import cv2 as cv

# Load the image
img = cv.imread('imgtemp/tablaR.jpg')

img = cv.GaussianBlur(img, (3,3), cv.BORDER_DEFAULT)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
canny = cv.Canny(gray, 125, 175)


blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)

cv.imshow('Lena - blur', blur)
cv.imshow('Lena - Canny', canny)

cv.waitKey(0)