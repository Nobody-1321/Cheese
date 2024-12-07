import cv2 as cv
import numpy as np
import cheese as che
def get_chess_cells_coords_from_file():
    coordinatess = {}

    with open('chess_table_cells.txt', 'r') as file:
        for line in file:
           # El primer paso es extraer el nombre (a8, b8, etc.) y las coordenadas
            line = line.strip()  # Eliminar espacios y saltos de línea innecesarios
            
           # Usamos una expresión regular para extraer el nombre y los arrays
            parts = line.split('(')
            name = parts[0]  # Nombre como 'a8', 'b8', etc.
            
            line = line.replace(name, '')
            line = line.replace('array', '')
            line = line.replace('(', '')
            line = line.replace(')', '')
            line = line.replace('[', '')
            line = line.replace(']', '')
            parts = line.split(',')


            # 8 valores de coordenadas (x, y) para cada celda
            coords = [(int(parts[i]), int(parts[i+1])) for i in range(0, len(parts), 2)]


            # Guardamos las coordenadas en un diccionario bajo su nombre
            coordinatess[name] = [np.array(coord) for coord in coords]

            
    return coordinatess

def get_roi_coords_from_file():
    with open('chess_table_roi.txt', 'r') as file:
        coords = file.read()
        coords = eval(coords)
    return coords

image = cv.imread('img/tabla2.jpg')
image = cv.resize(image, (560, 560))

chess_cells = get_chess_cells_coords_from_file()
roi_coords = get_roi_coords_from_file()

cchess_cells_normalized = {key.split('[array')[0]: value for key, value in chess_cells.items()}

coords  = cchess_cells_normalized['a8']

point1 = coords[0]
point2 = coords[2]

middle_point = che.middle_point(point1, point2)

che.draw_text(image, 'a8', middle_point, font_scale=0.5, thickness=1)

cv.imshow('image', image)
cv.waitKey(0)