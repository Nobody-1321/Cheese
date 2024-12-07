import numpy as np
from sklearn.cluster import KMeans
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

def classify_vertical_lines(lines, angle_threshold):
    
    thetas = lines[:, 0, 1]
    mask = np.abs(thetas) < angle_threshold
    classify_vertical_lines = lines[mask]

    return classify_vertical_lines

def classify_horizontal_lines(lines, angle_threshold):
    
    thetas = lines[:, 0, 1]
    mask = np.abs(thetas - np.pi/2) < angle_threshold

    classify_horizontal_lines = lines[mask]

    return classify_horizontal_lines


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
        print("No intersection point")
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

def degree_to_radian(degree):
    return degree * np.pi / 180

def radian_to_degree(radian):
    return radian * 180 / np.pi

def middle_point_line(line):
    x1, y1, x2, y2 = line
    return (x1 + x2) // 2, (y1 + y2) // 2

def middle_point(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return (x1 + x2) // 2, (y1 + y2) // 2

def cluster_and_lines(lines, n_clusters):
    middle_points = [middle_point_line(line) for line in lines]
    X = np.array(middle_points)
    
    # Aplicar K-Means con el número especificado de clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    
    print('Number of clusters: %d' % n_clusters)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, n_clusters)]
    
    representative_lines = []
    
    for i in range(n_clusters):
        centroid = centroids[i]
        
        # Encontrar la línea más cercana al centroide
        distances = [np.linalg.norm(np.array(middle_point_line(line)) - centroid) for line in lines]
        closest_line_index = np.argmin(distances)
        representative_lines.append(lines[closest_line_index])
    
    
    #representative_lines = [list(line) for line in representative_lines]
    representative_lines = np.array(representative_lines)
    return representative_lines