import cv2
import numpy as np

# Cargamos las dos imágenes
img1 = cv2.imread('roi/roi_0.png')
img2 = cv2.imread('roi/roi_2.png')

# Restamos las dos imágenes
resta = cv2.subtract(img1, img2)

# Mostramos el resultado
cv2.imshow('Resta', resta)
cv2.waitKey(0)
cv2.destroyAllWindows()