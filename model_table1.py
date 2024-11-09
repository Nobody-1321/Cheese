import cv2
import numpy as np
import cheese as che

imagen = cv2.imread('img/tabla.jpg')

imagen_escalada = cv2.resize(imagen, (0, 0), fx=0.8, fy=0.8)

grayimg = cv2.cvtColor(imagen_escalada, cv2.COLOR_BGR2GRAY) 

bordes = cv2.Canny(grayimg, 50, 150, apertureSize=3)

lineas = cv2.HoughLines(bordes, 1, np.pi/180, 200, min_theta=0, max_theta=np.pi)

bordes_color = cv2.cvtColor(bordes, cv2.COLOR_GRAY2BGR)

#filtrar lineas
lineas_filtradas = []


for linea in lineas:
    rho, theta = linea[0]
    agregar = True

    for linea_f in lineas_filtradas:
        if che.distancia(linea[0], linea_f[0]) < 20 and abs(theta - linea_f[0][1]) < 0.1:
            agregar = False
            break
    
    if agregar:
        lineas_filtradas.append(linea)

height, width = imagen_escalada.shape[:2]
max_length = int(np.hypot(width, height))


for linea in lineas_filtradas:
    bordes_colorc = bordes_color.copy()
    rho, theta = linea[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + max_length  * (-b))
    y1 = int(y0 + max_length  * (a))
    x2 = int(x0 - max_length  * (-b))
    y2 = int(y0 - max_length  * (a))
    cv2.line(imagen_escalada, (x1, y1), (x2, y2), (0, 0, 255), 2)


puntos_interseccion = []
for i in range(len(lineas_filtradas)):
    for j in range(i+1, len(lineas_filtradas)):
        puntos_interseccion.append(che.punto_interseccion(lineas_filtradas[i][0], lineas_filtradas[j][0]))

#eliminar puntos none

puntos_interseccion = [punto for punto in puntos_interseccion if punto is not None]

#ordenar puntos
puntos_interseccion_ordenados = che.ordenar_puntos(puntos_interseccion)

#for punto in puntos_interseccion_ordenados:
for i, punto in enumerate(puntos_interseccion_ordenados):
    cv2.putText(imagen_escalada, str(i), punto, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.imshow('lineas', imagen_escalada)

    cv2.waitKey(1)


filas = 8
columnas = 8

indices = che.obtener_indices(columnas, filas)

for indice in indices:
    punto1 = puntos_interseccion_ordenados[indice[0]]
    punto2 = puntos_interseccion_ordenados[indice[1]]
    punto3 = puntos_interseccion_ordenados[indice[2]]
    punto4 = puntos_interseccion_ordenados[indice[3]]
    
    # Definir el polígono usando los puntos de los vértices
    poligono = np.array([punto1, punto2, punto4, punto3], np.int32)  # Asegúrate del orden correcto
    
    # Dibujar el polígono lleno en la imagen
    cv2.fillConvexPoly(imagen_escalada, poligono, (0, 0, 255)) 

    cv2.imshow('lineas', imagen_escalada)
    cv2.waitKey(1)


#colocar texto en cada celda

for i, indice in enumerate(indices):
    
    
    punto1 = puntos_interseccion_ordenados[indice[0]]
    punto2 = puntos_interseccion_ordenados[indice[1]]
    punto3 = puntos_interseccion_ordenados[indice[2]]
    punto4 = puntos_interseccion_ordenados[indice[3]]
    x = (punto1[0] + punto2[0] + punto3[0] + punto4[0]) // 4
    y = (punto1[1] + punto2[1] + punto3[1] + punto4[1]) // 4
    
    #mostrar el indice de la celda
    #simulanado una matriz 9x9
    
    fila =  i // columnas
    columna = i % columnas

    cv2.putText(imagen_escalada, f"{str(fila)}:{str(columna)} ", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow('lineas', imagen_escalada)
    cv2.waitKey(10)


cv2.waitKey(0)