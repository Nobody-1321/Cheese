import cv2
import numpy as np
import cheese as che

# Cargar la imagen del tablero
imagen = cv2.imread('img/tabla.jpg')
imagen= cv2.resize(imagen, (0, 0), fx=0.8, fy=0.8)

# Detectar intersecciones
imagen_procesada, puntos_interseccion = che.encontrar_intersecciones_con_shi_tomasi(imagen)

#ordenar puntos
puntos_interseccion = puntos_interseccion.reshape(-1, 2)

puntos_ordenados = che.ordenar_puntos(puntos_interseccion)


indices = che.obtener_indices(8, 8)

for indice in indices:

    punto1 = tuple(np.int32(puntos_ordenados[indice[0]]))
    punto2 = tuple(np.int32(puntos_ordenados[indice[1]]))
    punto3 = tuple(np.int32(puntos_ordenados[indice[2]]))
    punto4 = tuple(np.int32(puntos_ordenados[indice[3]]))

    print(punto1, punto2, punto3, punto4)

    # Definir el polígono usando los puntos de los vértices
    poligono = np.array([punto1, punto2, punto4, punto3], np.int32)  # Asegúrate del orden correcto
    
    # Dibujar el polígono lleno en la imagen
    cv2.fillConvexPoly(imagen_procesada, poligono, (0, 0, 255)) 

    cv2.imshow('lineas', imagen_procesada)
    cv2.waitKey(1)


for i, indice in enumerate(indices):
    

    punto1 = puntos_ordenados[indice[0]]
    punto2 = puntos_ordenados[indice[1]]
    punto3 = puntos_ordenados[indice[2]]
    punto4 = puntos_ordenados[indice[3]]
    x = (punto1[0] + punto2[0] + punto3[0] + punto4[0]) // 4
    y = (punto1[1] + punto2[1] + punto3[1] + punto4[1]) // 4
    
    #mostrar el indice de la celda
    #simulanado una matriz 9x9
    
    fila =  i // 8
    columna = i % 8

    cv2.putText(imagen_procesada, f"{str(fila)}:{str(columna)} ", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow('lineas', imagen_procesada)
    cv2.waitKey(10)

cv2.waitKey(0)
