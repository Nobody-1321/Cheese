import cv2 
import numpy as np

def encontrar_intersecciones_con_shi_tomasi(imagen):
    maximo_puntos=100
    calidad=0.01
    min_distancia=10
    # Convertir la imagen a escala de grises
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Aplicar un desenfoque para reducir el ruido
    gris = cv2.GaussianBlur(gris, (5, 5), 0)
    
    # Detectar esquinas utilizando el detector de esquinas Shi-Tomasi
    esquinas = cv2.goodFeaturesToTrack(gris, maximo_puntos, calidad, min_distancia)
    
    # Verificar si se detectaron esquinas
    if esquinas is not None:
        esquinas = esquinas.astype(np.int32)
       

    return imagen, esquinas


#exportar la funcion

__all__ = ['encontrar_intersecciones_con_shi_tomasi']