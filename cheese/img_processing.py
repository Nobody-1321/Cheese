import cv2 as cv
import numpy as np

def open_image(path):
    """
    Open an image from a file path

    
    Parameters:
        path (str): The path to the image file

    Returns:
        image (np.array): The image as a numpy array,
        or None if the image could not be opened.
        image is in BGR format
    """
    return cv.imread(path)

def convert_to_gray(image):
    """
    Convert an image to grayscale

    Parameters:
        image (np.array): The image to convert
    
    Returns:
        gray_image (np.array): The grayscale image
    """

    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

def show_image_wait(window_name, image):
    """
    Show an image in a window and wait for a key press

    Parameters:
        window_name (str): The name of the window
        image (np.array): The image to show
    """

    cv.imshow(window_name, image)
    cv.waitKey(0)

def destroy_windows():
    """
    Destroy all windows created by show_image_wait 
    or show_image_wait_time
    """
    cv.destroyAllWindows()

def show_image_wait_time(window_name, image, time):
    """
    Show an image in a window and wait for a amount of time

    Parameters:
        window_name (str): The name of the window
        image (np.array): The image to show
        time (int): The time in milliseconds to wait
    """
    cv.imshow(window_name, image)
    cv.waitKey(time)

def denoise(image, kernel_size):
    return cv.GaussianBlur(image, (kernel_size, kernel_size), 0)

def detect_edges(image, threshold1, threshold2, aperture_size):
    """
    Detect edges in an image using the Canny algorithm

    Parameters:
        image (np.array): The image to process
        threshold1 (float): The first threshold for the hysteresis procedure
        threshold2 (float): The second threshold for the hysteresis procedure
        aperture_size (int): The aperture size for the Sobel operator
    
    Returns:
        edges (np.array): The edges detected in the image
    """

    return cv.Canny(image, threshold1, threshold2, aperture_size)

def detect_lines(image, rho, theta, threshold, min_theta, max_theta):
    """
    Detect lines in an image using the Hough transform.

    Parameters:
        image (np.array): The image to process
        rho (float): The resolution of the parameter r in pixels
        theta (float): The resolution of the parameter theta in radians
        threshold (int): The minimum number of intersections to detect a line
        min_theta (float): The minimum angle to detect a line
        max_theta (float): The maximum angle to detect a line

    Returns:
        lines (np.array): The lines detected in the image,
        the format is polar coordinates (rho, theta).
        
    """

    return cv.HoughLines(image, rho, theta, threshold, min_theta, max_theta)

def draw_lines(image, lines, color, thickness):
    """
    Draw lines in an image

    Parameters:
        image (np.array): The image to draw the lines
        lines (np.array): The lines to draw, the format is cartesian coordinates (x1, y1, x2, y2)
        color (tuple): The color of the lines (0, 0, 0)
        thickness (int): The thickness of the lines

    Returns:
        image (np.array): The image with the lines drawn
    """

    for line in lines:
        x1, y1, x2, y2 = line
        cv.line(image, (x1, y1), (x2, y2), color, thickness)
    
    return image

def draw_line(image, line, color, thickness):
    """
    Draw a line in an image

    Parameters:
        image (np.array): The image to draw the line
        line (np.array): The line to draw, the format is cartesian coordinates (x1, y1, x2, y2)
        color (tuple): The color of the line (0, 0, 0)
        thickness (int): The thickness of the line

    Returns:
        image (np.array): The image with the line drawn
    """
    x1, y1, x2, y2 = line
    return cv.line(image, (x1, y1), (x2, y2), color, thickness)

def draw_points(image, points, color, radius):
    """
    Draw points in an image

    Parameters:
        image (np.array): The image to draw the points
        points (np.array): The points to draw, the format is (x, y)
        color (tuple): The color of the points (0, 0, 0)
        radius (int): The radius of the points

    Returns:
        image (np.array): The image with the points drawn

    """
    for point in points:
        x, y = point
        cv.circle(image, (x, y), radius, color, -1)
    return image

def draw_point(image, point, color, radius):
    """
    Draw a point in an image

    Parameters:
        image (np.array): The image to draw the point
        point (np.array): The point to draw, the format is (x, y)
        color (tuple): The color of the point (0, 0, 0)
        radius (int): The radius of the point

    Returns:
        image (np.array): The image with the point drawn

    """  

    x, y = point
    return cv.circle(image, (x, y), radius, color, -1)

def draw_text(image, text, point, font_scale=1, color=(0, 0, 255), thickness=1):
    """
    Draw text in an image

    Parameters:
        image (np.array): The image to draw the text
        text (str): The text to draw
        point (tuple): The point to draw the text
        font_scale (float): The font scale
        color (tuple): The color of the text (0, 0, 0)
        thickness (int): The thickness of the text

    Returns:
        image (np.array): The image with the text drawn

    """    
    return cv.putText(image, text, point, cv.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def draw_polygon(image, polygon, color, thickness):
    """
    Draw a polygon in an image

    Parameters:
        image (np.array): The image to draw the polygon
        polygon (np.array): The polygon to draw, the format is a list of points [(x1, y1), (x2, y2), ...]
        color (tuple): The color of the polygon (0, 0, 0)
        thickness (int): The thickness of the polygon

    Returns:
        image (np.array): The image with the polygon drawn

    """
    polygon = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv.polylines(image, [polygon], isClosed=True, color=color, thickness=thickness)
    
    return image

def capture_video(camera_index, width, height, fps):
    """
    Capture video from a camera 

    Parameters:
        camera_index (int): The index of the camera
        width (int): The width of the video
        height (int): The height of the video
        fps (int): The frames per second of the video

    Returns:
        cap (cv.VideoCapture): The video capture object

    """

    cap = cv.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("cannot open camera")
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv.CAP_PROP_FPS, fps)
    return cap

def resize_image(image, width, height):
    """
    Resize an image

    Parameters:
        image (np.array): The image to resize
        width (int): The width of the new image
        height (int): The height of the new image

    Returns:
        image (np.array): The resized image

    """

    return cv.resize(image, (width, height))

def draw_roi(image, roi_coords, color, thickness):
    
    x1, y1, x2, y2 = map(int, [roi_coords["x1"], roi_coords["y1"], roi_coords["x2"], roi_coords["y2"]])
    
    if x1 < 0  or x1 > image.shape[1]:
        return image
    if x2 < 0  or x2 > image.shape[1]:
        return image
    if y1 < 0  or y1 > image.shape[0]:
        return image
    if y2 < 0  or y2 > image.shape[0]:
        return image
    
    #dibujar solo las coordnas son diferentes
    if x1 != x2 and y1 != y2:
        cv.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    return image
    
def save_image(path, image):
    """
    Save an image to a file path the image must be in (0,0,0) format

    Parameters:
        path (str): The path to save the image
        image (np.array): The image to save
    """

    if image.shape[0] > 0 and image.shape[1] > 0 and image.shape[2] == 3:
        cv.imwrite(path, image)
    else:
        print("No se pudo guardar la imagen")
          
def extract_roi(image, roi_coords):
    """
    Extract a region of interest from an image 

    Parameters:
        image (np.array): The image to extract the region of interest
        roi_coords (dict): The coordinates of the region of interest, the format is {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

    Returns:
        roi (np.array): The region of interest extracted from the image
    """

    x1, y1, x2, y2 = map(int, [roi_coords["x1"], roi_coords["y1"], roi_coords["x2"], roi_coords["y2"]])
    return image[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]

def extract_cell(image, cell):
    """
    Extract a cell from an image

    Parameters:
        image (np.array): The image to extract the cell
        cell (np.array): The cell to extract, the format is [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

    Returns:
        cell (np.array): The cell extracted from the image
    """
    x1, y1 = cell[0]
    x3, y3 = cell[2]
    return image[y1:y3, x1:x3]

def extract_chess_cell(image, cells, coordinate):
    """
    Extract a cell from an image using chess coordinates

    Parameters:
        image (np.array): The image to extract the cell
        cells (dict): The cells of the chessboard, the format is {"a1": cell1, "a2": cell2, ...}
        coordinate (str): The chess coordinate of the cell to extract

    Returns:
        cell (np.array): The cell extracted from the image
    """
    return extract_cell(image, cells[coordinate])

def ecualizacion_histograma_adaptativo(imagen):
    """
    Realiza la ecualización del histograma adaptativo (CLAHE) sobre una imagen.
    
    :param imagen: Imagen de entrada en formato numpy array (debe estar en escala de grises).
    :return: Imagen con ecualización del histograma adaptativo.
    """
    # Asegurarse de que la imagen está en escala de grises
    if len(imagen.shape) != 2:
        raise ValueError("La imagen debe estar en escala de grises.")

    # Crear el objeto CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))

    # Aplicar CLAHE a la imagen
    imagen_ecualizada = clahe.apply(imagen)

    return imagen_ecualizada

def ecualizacion_histograma(imagen):
    """
    Realiza la ecualización del histograma sobre una imagen.
    
    :param imagen: Imagen de entrada en formato numpy array (debe estar en escala de grises).
    :return: Imagen con ecualización del histograma.
    """
    # Asegurarse de que la imagen está en escala de grises
    if len(imagen.shape) != 2:
        raise ValueError("La imagen debe estar en escala de grises.")

    # Aplicar la ecualización del histograma a la imagen
    imagen_ecualizada = cv.equalizeHist(imagen)

    return imagen_ecualizada

def high_pass_filter(image, kernel_size):
    """
    Apply a high pass filter to an image

    Parameters:
        image (np.array): The image to apply the filter
        kernel_size (int): The size of the kernel

    Returns:
        image (np.array): The image with the filter applied
    """

    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    return cv.filter2D(image, -1, kernel)

def laplacian_filter(image):
    """
    Apply a laplacian filter to an image

    Parameters:
        image (np.array): The image to apply the filter

    Returns:
        image (np.array): The image with the filter applied
    """

    laplacian = cv.Laplacian(image, cv.CV_64F)
    return cv.convertScaleAbs(laplacian)

def gamma_correction(image, gamma):
    """
    Apply gamma correction to an image

    Parameters:
        image (np.array): The image to apply the correction
        gamma (float): The gamma value

    Returns:
        image (np.array): The image with the correction applied
    """
    img_normalized = image / 255.0

    img_gamma_corrected = np.array(255 * (img_normalized ** gamma), dtype=np.uint8)
    return img_gamma_corrected

def fastNlMeansDenoising (image, h=3, templateWindowSize=7, searchWindowSize=21):
    """
    Apply a fast non-local means denoising filter to an image

    Parameters:
        image (np.array): The image to apply the filter
        h (int): The strength of the filter
        templateWindowSize (int): The size of the window for the template
        searchWindowSize (int): The size of the window for the search

    Returns:
        image (np.array): The image with the filter applied
    """

    return cv.fastNlMeansDenoising(image, h, templateWindowSize, searchWindowSize)

def bilateralFilter(image, d, sigmaColor, sigmaSpace):
    """
    Apply a bilateral filter to an image

    Parameters:
        image (np.array): The image to apply the filter
        d (int): The diameter of each pixel neighborhood
        sigmaColor (int): The filter sigma in the color space
        sigmaSpace (int): The filter sigma in the coordinate space

    Returns:
        image (np.array): The image with the filter applied
    """

    return cv.bilateralFilter(image, d, sigmaColor, sigmaSpace)

def sitf_features(image,sitf):
    keypoints, descriptors = sitf.detectAndCompute(image, None)
    return keypoints, descriptors

def feature_matching(descriptors1, descriptors2, min_distance):
    """
    Match features between two images using the Brute-Force Matcher

    Parameters:
        descriptors1 (np.array): The descriptors of the first image
        descriptors2 (np.array): The descriptors of the second image
        min_distance (float): The minimum distance to consider a match

    Returns:
        matches (np.array): The matches between the features of the two images
    """
    if descriptors1 is None or descriptors2 is None:
        return False

    descriptors1 = descriptors1.astype('float32')
    descriptors2 = descriptors2.astype('float32')


    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = []
    
    for m in matches:
        if len(m) == 2:
            m1, m2 = m
            if m1.distance < min_distance * m2.distance:
                good_matches.append(m1)
        
    
    if len(good_matches) > 4:
        return True
    else:
        return False
    
def extract_black_regions(image, threshold):
    _, new_image = cv.threshold(image, threshold, 255, cv.THRESH_BINARY)
    return new_image

def apply_mask(image, mask):
    return cv.bitwise_and(image, image, mask=mask)

def background_features_chess_board(image, chess_cells, sitf):
    #show_image_wait('por',image)
    features = {}
    for cell_name in chess_cells:
        cell = extract_chess_cell(image, chess_cells, cell_name)
        keypoint, descriptor = sitf_features(cell, sitf)
        features[cell_name] = (keypoint, descriptor)

    return features

def extract_gray_regions(image, threshold1, threshold2):
    _, new_image = cv.threshold(image, threshold1, threshold2, cv.THRESH_BINARY)
    return new_image

def model_features_chess_board(image, chess_cells, cell_name, sitf):
    features = {}
    cell = extract_chess_cell(image, chess_cells, cell_name)
    keypoint, descriptor = sitf_features(cell, sitf)
    features[cell_name] = (keypoint, descriptor)
    return features

def models_features_chess_board(roi, chess_cells, sitf):
    features = {}
    for cell_name in chess_cells:
        cell = extract_chess_cell(roi, chess_cells, cell_name)
        mask = extract_black_regions(cell, 50)
        cell = apply_mask(cell, mask)
        mask = extract_gray_regions(cell, 65, 120)
        cell = apply_mask(cell, mask)
 #       show_image_wait('por',cell)
#        destroy_windows()
        keypoint, descriptor = sitf_features(cell, sitf)
        features[cell_name] = (keypoint, descriptor)

    return features

def drawn_cheese_table(image, cell, text):
    x1, y1 = cell[0]
    x3, y3 = cell[2]
    middle_x = (x1 + x3) // 2
    middle_y = (y1 + y3) // 2
    #to int 
    middle_x = int(middle_x)
    middle_y = int(middle_y)
    return draw_text(image, text, (middle_x, middle_y), 1.5)

def search_model_features_background(background_feactures, model_features, cell_name_model , chess_cells):
    cell_res = None
    model_points, model_descriptors = model_features[cell_name_model]
    
    for cell_name in chess_cells:
        back_cell, back_descriptors = background_feactures[cell_name] 
        res= feature_matching(model_descriptors, back_descriptors, 0.7)
        if res:
            cell_res = cell_name
        else:
            print("No se encontro la celda")
            continue

    return cell_res        