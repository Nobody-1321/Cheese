import cv2
import numpy as np

# Cargar la imagen de entrada y la plantilla
#imagen = cv2.imread('img/tabla.jpg')
imagen = cv2.imread('imgtemp/Multimedia2.jpeg')
imagen = cv2.resize(imagen, (0, 0), fx=0.4, fy=0.4)

#plantilla = cv2.imread('imgtemp/Multimedia2.jpeg')
plantilla = cv2.imread('img/tabla2.jpg')
plantilla = cv2.resize(plantilla, (0, 0), fx=0.7, fy=0.7)

# Convertir a escala de grises (opcional, dependiendo del caso)
imagen_gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
plantilla_gray = cv2.cvtColor(plantilla, cv2.COLOR_BGR2GRAY)

# Hacer Template Matching
resultado = cv2.matchTemplate(imagen_gray, plantilla_gray, cv2.TM_CCOEFF_NORMED)

# Obtener las coordenadas donde se encuentra la mejor coincidencia
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(resultado)

# Dibujar un rectángulo alrededor de la mejor coincidencia
top_left = max_loc
bottom_right = (top_left[0] + plantilla.shape[1], top_left[1] + plantilla.shape[0])
cv2.rectangle(imagen, top_left, bottom_right, (0, 255, 0), 2)

# Mostrar la imagen con el rectángulo
cv2.imshow('Resultado', imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()
