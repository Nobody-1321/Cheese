import numpy as np
from sklearn.cluster import DBSCAN 
import matplotlib.pyplot as plt


def classify_polar_lines(lines, angle_threshold):
    vertical_lines = []
    horizontal_lines = []

    for line in lines:
        for rho, theta in line:
            if abs(theta - 0) < angle_threshold or abs(theta - np.pi) < angle_threshold:
                horizontal_lines.append(line)
            elif abs(theta - np.pi/2) < angle_threshold:
                vertical_lines.append(line)
    return vertical_lines, horizontal_lines

def line_distance(line1, line2):
    rtho1, theta1 = line1
    rtho2, theta2 = line2
    return abs(rtho1 - rtho2)

def filter_close_lines(lines, distance_threshold, angle_threshold):

    filtered_lines = []

    for line in lines:
        rho, theta = line[0]
        add = True

        for filtered_line in filtered_lines:
            if line_distance(line[0], filtered_line[0]) < distance_threshold and abs(theta - filtered_line[0][1]) < angle_threshold:
                add = False
                break
        if add:
            filtered_lines.append(line)

        add = False

    return filtered_lines

def polar_to_cartesian(line, max_length):
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + max_length * (-b))
    y1 = int(y0 + max_length * (a))
    x2 = int(x0 - max_length * (-b))
    y2 = int(y0 - max_length * (a))
    return x1, y1, x2, y2

def intersection_point(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if det != 0:
        x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / det
        y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / det
        return (int(x), int(y))
    else:
        return None
    
'''
def intersection_point(line1, line2):
    rho1, theta1 = line1
    rho2, theta2 = line2

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
'''

def filter_close_points(points, distance_threshold):
    filtered_points = []

    for point in points:
        add = True

        for filtered_point in filtered_points:
            if np.linalg.norm(np.array(point) - np.array(filtered_point)) < distance_threshold:
                add = False
                break
        if add:
            filtered_points.append(point)
    return filtered_points

def sort_points(points):
    points_sorted_x = sorted(points, key=lambda point: (point[1], point[0]))

    sorted_intersection_points = []

    for i in range(0, len(points_sorted_x), 9):
        row = points_sorted_x[i:i+9]
        sorted_row = sorted(row, key=lambda point: point[0])
        sorted_intersection_points.extend(sorted_row)

    return sorted_intersection_points

def get_indices(columns, rows):
    indices = []
    for i in range(columns):
        k1 = i * (rows + 1)
        k2 = k1 + rows + 1

        for j in range(rows):
           indices.append((k1 + j, k1 + j + 1, k2 + j, k2 + j + 1))

    return indices

def process_roi_coords(roi_coords, image_width, image_height):
    x1, y1, x2, y2 = map(int, [roi_coords["x1"], roi_coords["y1"], roi_coords["x2"], roi_coords["y2"]])
    
    x1 = max(0, min(x1, image_width))
    x2 = max(0, min(x2, image_width))
    y1 = max(0, min(y1, image_height))
    y2 = max(0, min(y2, image_height))

    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

def get_roi(frame, roi_coords):
    x1, y1, x2, y2 = map(int, [roi_coords["x1"], roi_coords["y1"], roi_coords["x2"], roi_coords["y2"]])
    return frame[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)] 

def point_into_cell(point, cell):
    x, y = point
    x1, y1 = cell[0]
    x2, y2 = cell[1]
    x3, y3 = cell[2]
    x4, y4 = cell[3]

    if x1 <= x <= x2 and y1 <= y <= y4:
        return True
    return False

def middle_point_line(line):
    x1, y1, x2, y2 = line
    return (x1 + x2) // 2, (y1 + y2) // 2

def cluster_and_average_lines(lines, distance_threshold, angle_threshold):
    """
    Agrupa líneas cercanas usando DBSCAN y promedia las líneas en cada cluster.
    
    Args:
        lines (list): Lista de líneas en formato [(rho, theta)].
        distance_threshold (float): Máxima distancia para considerar líneas cercanas (rho).
        angle_threshold (float): Máxima diferencia de ángulo (en radianes) para agrupar líneas.
    
    Returns:
        list: Lista de líneas promediadas después de agruparlas.
    """
    
    #lines = np.array(lines)
    
    # Convertir las líneas a una matriz para DBSCAN
    lines_array = []
    for line in lines:
        rho, theta = line[0]
        lines_array.append([rho, theta])
        print(rho)
    
    lines_array = np.array(lines_array)
    
    if lines_array.ndim != 2 or lines_array.shape[1] != 2:
        raise ValueError("Expected 2D array with shape (n_samples, 2)")

    # Ajustar los parámetros de DBSCAN
    eps = np.sqrt(distance_threshold**2 + angle_threshold**2)  # Radio de agrupamiento
    clustering = DBSCAN(eps=eps, min_samples=1).fit(lines_array)
    
    
    # Agrupar y promediar líneas
    averaged_lines = []
    for cluster_id in np.unique(clustering.labels_):
        # Filtrar líneas del cluster actual
        cluster_lines = lines_array[clustering.labels_ == cluster_id]
        
        # Calcular el promedio de rho y theta
        avg_rho = np.mean(cluster_lines[:, 0])
        avg_theta = np.mean(cluster_lines[:, 1])
        
        # Agregar línea promedio a la lista
        averaged_lines.append((avg_rho, avg_theta))
    #para de array numpy a lista
    for i in range(len(averaged_lines)):
        print(averaged_lines[i])

    #mostrar las lineas promediadas con matplot 
    plt.figure()
    ax = plt.gca()
    for line in averaged_lines:
        rho, theta = line
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        ax.plot([x1, x2], [y1, y2], 'r')

    plt.show()
    

    return  averaged_lines