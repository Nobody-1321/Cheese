import cv2 as cv
import numpy as np

def open_image(path):
    return cv.imread(path)

def convert_to_gray(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

def show_image_wait(window_name, image):
    cv.imshow(window_name, image)
    cv.waitKey(0)

def destroy_windows():
    cv.destroyAllWindows()

def show_image_wait_time(window_name, image, time):
    cv.imshow(window_name, image)
    cv.waitKey(time)

def denoise(image, kernel_size):
    return cv.GaussianBlur(image, (kernel_size, kernel_size), 0)

def detect_edges(image, threshold1, threshold2, aperture_size):
    return cv.Canny(image, threshold1, threshold2, aperture_size)

def detect_lines(image, rho, theta, threshold, min_theta, max_theta):
    return cv.HoughLines(image, rho, theta, threshold, min_theta, max_theta)

def draw_lines(image, lines, color, thickness):
    image_ = image.copy()
    for line in lines:
        x1, y1, x2, y2 = line
        cv.line(image_, (x1, y1), (x2, y2), color, thickness)
    return image_

def draw_line(image, line, color, thickness):
    x1, y1, x2, y2 = line
    return cv.line(image, (x1, y1), (x2, y2), color, thickness)

def draw_points(image, points, color, radius):
    image_ = image.copy()
    for point in points:
        x, y = point
        cv.circle(image_, (x, y), radius, color, -1)
    return image_

def draw_point(image, point, color, radius):
    x, y = point
    return cv.circle(image, (x, y), radius, color, -1)

def draw_text(image, text, point, font_scale=1, color=(0, 0, 0), thickness=1):
    return cv.putText(image, text, point, cv.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def draw_polygon(image, polygon, color, thickness):
    # Convertir el polígono al formato adecuado
    polygon = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv.polylines(image, [polygon], isClosed=True, color=color, thickness=thickness)
    
    return image

def capture_video(camera_index, width, height, fps):
    cap = cv.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("cannot open camera")
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv.CAP_PROP_FPS, fps)
    return cap

def get_frame(cap):
    ret, frame = cap.read()
    if ret:
        return frame
    return None

def get_frame_tex(cap):
    ret, frame = cap.read()
    if ret:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGBA)
        frame_data = frame.flatten().astype(np.float32) / 255.0
        return frame_data
    return None

def resize_image(image, width, height):
    return cv.resize(image, (width, height))

