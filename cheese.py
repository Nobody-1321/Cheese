import cheese as che
import numpy as np
import cv2 
from tensorflow.keras import models # type: ignore

roi_coords = {"x1": 0, "y1": 0, "x2": 0, "y2": 0}
drawing = False  # Indica si el usuario está dibujando el rectángulo
cells = []
chess_cells = []
black_cells = {}
white_cells = {}
index = 0
k=0
l =64

def draw_rectangle(event, x, y, flags, param):
    global roi_coords, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        # Cuando el botón izquierdo del mouse se presiona, inicia el rectángulo
        drawing = True
        roi_coords["x1"], roi_coords["y1"] = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        # Mientras el mouse se mueve con el botón izquierdo presionado, actualiza la segunda esquina
        roi_coords["x2"], roi_coords["y2"] = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        # Cuando se suelta el botón izquierdo, termina el dibujo del rectángulo
        drawing = False
        roi_coords["x2"], roi_coords["y2"] = x, y

def get_roi_coords():
    global roi_coords, drawing, cells, chess_cells

    print('Getting chess table...')

    cap = che.capture_video(2, 640, 480, 30)
    
    if not cap.isOpened():
        print("Error: No se puede abrir la cámara")
        exit()

    cv2.namedWindow("Webcam")
    cv2.setMouseCallback("Webcam", draw_rectangle)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: No se puede recibir frame (stream end?). Saliendo ...")
            break

        if drawing or (roi_coords["x1"] != roi_coords["x2"] and roi_coords["y1"] != roi_coords["y2"]):
            x1, y1, x2, y2 = roi_coords["x1"], roi_coords["y1"], roi_coords["x2"], roi_coords["y2"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv2.imshow('Webcam', frame)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break
        
        elif key == ord('s') and not drawing:
            i = 0
            roi = che.extract_roi(frame, roi_coords)
            che.save_image('chess_table.png', roi)
            image = che.open_image('chess_table.png')
            image = che.resize_image(image, 560, 560)
            
            image = che.denoise(image, 9)
            

            image_ = che.convert_to_gray(image) 
            image_ = che.detect_edges(image_, 1, 130, 3)
            che.show_image_wait('edges', image_)
            lines = che.detect_lines(image_, 1, che.np.pi/180, 100, 0, che.np.pi)
            copy_image = image.copy()
            
            for line in lines:
                line = che.polar_to_cartesian(line, 1000)
                copy_image = che.draw_line(copy_image, line, (0, 0, 255), 2)
            
            che.show_image_wait('lines', copy_image)

            horizontal_lines = che.classify_horizontal_lines(lines, che.degree_to_radian(20))
            vertical_lines = che.classify_vertical_lines(lines, che.np.pi/4)
            height, width = image.shape[:2]
            max_length = int(che.np.hypot(width, height))

            horizontal_lines = che.filter_close_lines(horizontal_lines, 30, che.np.pi/4)
            vertical_lines = che.filter_close_lines(vertical_lines, 30, che.np.pi/4)
            
            print(len(horizontal_lines))
            print(len(vertical_lines))

            cartesian_vertical_lines = np.array([che.polar_to_cartesian(line, max_length) for line in vertical_lines])
            cartesian_horizontal_lines = np.array([che.polar_to_cartesian(line, max_length) for line in horizontal_lines])
            cartesian_vertical_lines = che.cluster_and_lines(cartesian_vertical_lines, 9)
            cartesian_horizontal_lines = che.cluster_and_lines(cartesian_horizontal_lines, 9)

            for line in cartesian_vertical_lines:
                image = che.draw_line(image, line, (0, 0, 255), 2)
            
            for line in cartesian_horizontal_lines:
                image = che.draw_line(image, line, (0, 0, 255), 2)

            intersection_points = []

            for vertical_line in cartesian_vertical_lines:
                for horizontal_line in cartesian_horizontal_lines:
                    intersection_points.append(che.intersection_point(vertical_line, horizontal_line))
            
            intersection_points = [point for point in intersection_points if point is not None]

            intersection_points = np.array(intersection_points)

            intersection_points = che.sort_points(intersection_points)

            for point in intersection_points:
                image = che.draw_text(image, str(i), point, 0.5, (255, 0, 0), 2)
                i+=1
            
            che.show_image_wait('points', image)

            if cv2.waitKey(0) == ord('r'):
                cv2.destroyWindow('points')
                continue
            
            indices = che.get_indices(8, 8)

            for index in indices:
                point1 = intersection_points[index[0]]
                point2 = intersection_points[index[1]]
                point3 = intersection_points[index[2]]
                point4 = intersection_points[index[3]]


                cells.append((point1, point2, point4, point3))
                po1 = np.array([point1, point2, point4, point3], np.int32)
                if po1.shape[0] == 4:            
                    po1 = po1.reshape((-1, 1, 2))
                    image = che.draw_polygon(image, po1, (0, 255, 0), 2)
                else:
                    print('Error')
                    print(po1.shape[0])
                    print(po1)

            che.show_image_wait('lines', image)

            chess_cells = che.chess_coordinate_cells(cells, reverse=True)

            coodinate = 'e3'
            new_image = che.extract_chess_cell(image, chess_cells, coodinate)
            che.show_image_wait('cell', new_image)
            
            #exportar las coordenadas de las celdas  and roi_coords en un archivo txt
            
            with open('chess_cells.txt', 'w') as file:
                for cell in chess_cells:
                    file.write(str(cell) + str(chess_cells[cell]) + '\n')
            
            with open('roi_coords.txt', 'w') as file:
                file.write(str(roi_coords))
            

    cap.release()
    cv2.destroyAllWindows()

    return roi_coords

def get_roi_coords_from_file():
    with open('roi_coords.txt', 'r') as file:
        coords = file.read()
        coords = eval(coords)
    return coords

def get_chess_cells_coords_from_file(path):
    coordinatess = {}

    with open(path, 'r') as file:
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

def cheese_main():
    global index, k,l
    
    model = models.load_model('chess_model.h5')
    image_chess = cv2.imread('img/tabla2.jpg')
    image_chess = cv2.resize(image_chess, (560, 560))
    
    #coordinates = get_roi_coords()
    coordinates = get_roi_coords_from_file()
    chess_cells = get_chess_cells_coords_from_file('chess_cells.txt')
    all_chess_cells = che.all_chess_coordinates()
    chess_cells_normalized = {key.split('[array')[0]: value for key, value in chess_cells.items()}   

    cap = che.capture_video(2, 640, 480, 30)
    ret, frame = cap.read()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se puede recibir frame (stream end?). Saliendo ...")
            break

        roi = che.extract_roi(frame, coordinates)
        roi = che.resize_image(roi, 560, 560)        
        che.show_image_wait_time('roi', roi, 1)    

        for i in range(63):
            cell = chess_cells_normalized[all_chess_cells[i]]
            point1 = cell[0]
            point2 = cell[2]

            middle_point = che.middle_point(point1, point2)
            cell_img = che.extract_chess_cell(roi, chess_cells, all_chess_cells[i])
            cell_img = che.resize_image(cell_img, 85, 85)
            cell_img = np.expand_dims(cell_img, axis=0)
            cell_img = cell_img/255
            prediction = model.predict(cell_img)
            prediction = np.argmax(prediction)
           
            if prediction == 1:
                che.draw_text(image_chess, 'P',middle_point , 0.5, (0, 0, 255), 2)

        che.show_image_wait_time('chess', image_chess, 1)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break        
        
if __name__ == '__main__':
    cheese_main()

