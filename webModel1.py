import cv2
import numpy as np
import cheese as che

# Inicializar la captura de video desde la webcam (0 es el índice de la cámara por defecto)
cap = cv2.VideoCapture(2)


if not cap.isOpened():
    print("Error: No se puede abrir la cámara")
    exit()

lineas_filtradas = []



while True:
    # Capturar frame por frame
    ret, frame = cap.read()

    # Si no se pudo capturar el frame, salir del bucle
    if not ret:
        print("Error: No se puede recibir frame (stream end?). Saliendo ...")
        break
    framecolor = frame.copy()
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.Canny(frame, 50, 150, apertureSize=3)
    lineas = cv2.HoughLines(frame, 1, np.pi/180, 110, min_theta=0, max_theta=np.pi)
    #eliminar nonetype de lineas


    lineas_filtradas = []

    if lineas is not None:
        for linea in lineas:
            rho, theta = linea[0]
            agregar = True

            for linea_f in lineas_filtradas:
                if che.distancia(linea[0], linea_f[0]) < 50 and abs(theta - linea_f[0][1]) < 0.1:
                    agregar = False
                    break
            
            for linea_f in lineas_filtradas:
                if che.distancia(linea[0], linea_f[0]) < 50 and abs(theta - linea_f[0][1]) < 0.1:
                    agregar = False
                    break

            if agregar:
                lineas_filtradas.append(linea)

    
    print(len(lineas_filtradas))
    # Dibujar las líneas en el frame si hay más de 2 líneas filtradas
    if lineas_filtradas and len(lineas_filtradas) > 2:
        framecolor = che.dibujar_lineas(framecolor, lineas_filtradas)
        
    puntos_interseccion = []
    if len(lineas_filtradas) > 2:
        for i in range(len(lineas_filtradas)):
            for j in range(i+1, len(lineas_filtradas)):
                puntos_interseccion.append(che.punto_interseccion(lineas_filtradas[i][0], lineas_filtradas[j][0]))

    puntos_interseccion = [punto for punto in puntos_interseccion if punto is not None]
    puntos_interseccion = [punto for punto in puntos_interseccion if punto[0] > 0 and punto[1] > 0]
    puntos_interseccion = [punto for punto in puntos_interseccion if punto[0] < 631 and punto[1] < 470]

    puntos_interseccion_ordenados = che.ordenar_puntos(puntos_interseccion)

    #salvar los puntos de interseccion

    for i,punto in enumerate(puntos_interseccion_ordenados):
        cv2.putText(framecolor, f'{i}', tuple(punto), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    filas = 8
    columnas = 8

    indices = che.obtener_indices(filas, columnas)

    if len(puntos_interseccion_ordenados) == 81:
        for indice in indices:
            punto1 = puntos_interseccion_ordenados[indice[0]]
            punto2 = puntos_interseccion_ordenados[indice[1]]
            punto3 = puntos_interseccion_ordenados[indice[2]]
            punto4 = puntos_interseccion_ordenados[indice[3]]
    
    # Definir el polígono usando los puntos de los vértices
            poligono = np.array([punto1, punto2, punto4, punto3], np.int32)  # Asegúrate del orden correcto
    
    # Dibujar el polígono lleno en la imagen
            cv2.fillConvexPoly(framecolor, poligono, (0, 0, 255)) 


#colocar texto en cada celda

        print(len(puntos_interseccion_ordenados))

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

            cv2.putText(framecolor, f"{str(fila)}:{str(columna)} ", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)



    cv2.imshow('frame', frame)
    cv2.imshow('framecolor', framecolor)


    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Liberar la captura de video y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()