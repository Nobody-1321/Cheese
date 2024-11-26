import cv2
import numpy as np

def segmentar_por_color(imagen_path, color_min, color_max):
    """
    Segmenta una imagen basada en un rango de color en el espacio HSV.

    :param imagen_path: Ruta de la imagen a procesar.
    :param color_min: Rango mínimo del color en HSV (array o tupla).
    :param color_max: Rango máximo del color en HSV (array o tupla).
    """
    # Cargar la imagen
    imagen = cv2.imread("./img/caballo.jpeg")
    if imagen is None:
        print("Error: No se pudo cargar la imagen.")
        return

    # Convertir la imagen al espacio de color HSV
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

    # Crear una máscara basada en el rango de color
    mascara = cv2.inRange(hsv, np.array(color_min), np.array(color_max))

    # Aplicar la máscara a la imagen original
    resultado = cv2.bitwise_and(imagen, imagen, mask=mascara)

    # Mostrar los resultados
    cv2.imshow("Imagen Original", imagen)
    cv2.imshow("Mascara", mascara)
    cv2.imshow("Segmentacion por Color", resultado)

    # Esperar a que se cierre la ventana
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Ruta de la imagen
    ruta_imagen = "imagen_ejemplo.jpg"

    # Definir el rango de color en HSV
    # Por ejemplo, para segmentar un color blanco
    color_min_hsv = [0, 0, 200]  # Mínimo (H, S, V)
    color_max_hsv = [179, 50, 255]  # Máximo (H, S, V)

    # Llamar a la función de segmentación
    segmentar_por_color(ruta_imagen, color_min_hsv, color_max_hsv)
