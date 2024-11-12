import cv2 
import numpy as np


def dibujar_lineas(frame, lineas_filtradas):
    height, width = frame.shape[:2]
    max_length = int(np.hypot(width, height))
    for linea in lineas_filtradas:
        framec = frame.copy()
        rho, theta = linea[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + max_length  * (-b))
        y1 = int(y0 + max_length  * (a))
        x2 = int(x0 - max_length  * (-b))
        y2 = int(y0 - max_length  * (a))
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 8)
    return frame


__all__ = ['dibujar_lineas']