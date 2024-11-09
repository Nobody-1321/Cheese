import numpy as np

def distancia(linea1, linea2):
    rtho1, theta1 = linea1
    rtho2, theta2 = linea2
    return abs(rtho1 - rtho2)


def punto_interseccion(linea1, linea2):
    rho1, theta1 = linea1
    rho2, theta2 = linea2

    A1 = np.cos(theta1)
    B1 = np.sin(theta1)
    C1 = rho1

    A2 = np.cos(theta2)
    B2 = np.sin(theta2)
    C2 = rho2

    det = A1 * B2 - A2 * B1
    dx = C1 * B2 - C2 * B1
    dy = A1 * C2 - A2 * C1

    if det != 0:
        x = dx / det
        y = dy / det
        return (int(x), int(y))
    else:
        return None

__all__ = ['distancia', 'punto_interseccion']