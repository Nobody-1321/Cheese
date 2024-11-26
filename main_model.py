import cheese as che
import numpy as np
import cv2 as cv
def main():
    
    #image = che.open_image('imgtemp/tablaR2.jpg')
    image = che.open_image('img/tablaRea.jpg')
    #image = che.open_image('img/tablaRea1.jpg')
    #image = che.open_image('img/tablaRea2.jpg')
    #redimensionar a 560 560    
    image = cv.resize(image, (560, 560))

    image = che.denoise(image, 9)
    gray_image = che.convert_to_gray(image)

    che.show_image_wait('gray', gray_image)

    edeges = che.detect_edges(gray_image, 40, 140, 3)
    che.show_image_wait('edges', edeges)

    lines = che.detect_lines(edeges, 1, che.np.pi/180, 100, 0, che.np.pi)

    horizontal_lines, vertical_lines = che.classify_polar_lines(lines, che.np.pi/4)
    
    print(f'horizontal lines: {len(horizontal_lines)}')
    print(f'vertical lines: {len(vertical_lines)}')
    print(f'angle: {che.np.pi/4}')
    

    #exit()
    vertical_lines = che.filter_close_lines(vertical_lines, 60, che.np.pi/4)
    horizontal_lines = che.filter_close_lines(horizontal_lines, 60, che.np.pi/4)

    for line in vertical_lines:
        
        linea_cartesiana = che.polar_to_cartesian(line, 791)
        punto_medio = che.middle_point_line(linea_cartesiana)
        image = che.draw_line(image, linea_cartesiana, (0, 0, 255), 2)
        
        che.show_image_wait_time('lines', image, 1000)

        for rho, theta in line:
            print(f'rho: {rho}, theta: {theta}')
            print(linea_cartesiana)
            print(punto_medio)

    che.show_image_wait('lines', image)
    height, width = image.shape[:2]
    max_length = int(che.np.hypot(width, height))

    print(f'max_length: {max_length}')
    exit()

    cartesian_vertical_lines = [che.polar_to_cartesian(line, max_length) for line in vertical_lines]
    cartesian_horizontal_lines = [che.polar_to_cartesian(line, max_length) for line in horizontal_lines]

    print(len(cartesian_vertical_lines))
    print(len(cartesian_horizontal_lines))

    for line in cartesian_vertical_lines:
        image = che.draw_line(image, line, (0, 0, 255), 2)
        #che.show_image_wait_time('lines', image, 1000)


    for line in cartesian_horizontal_lines:
        image = che.draw_line(image, line, (0, 0, 255), 2)
        #che.show_image_wait_time('lines', image, 1000)

    che.show_image_wait('lines', image)

    intersection_points = []
    
    for vertical_line in cartesian_vertical_lines:
        for horizontal_line in cartesian_horizontal_lines:
            intersection_points.append(che.intersection_point(vertical_line, horizontal_line))

    print(len(intersection_points))

    intersection_points = [point for point in intersection_points if point is not None]
    intersection_points = che.sort_points(intersection_points)

    intersection_points = che.filter_close_points(intersection_points, 30) 

    print(len(intersection_points))

    i=0
    for point in intersection_points:
        imgae = che.draw_text(image, str(i), point, 0.5, (255, 0, 0), 2)
        #che.show_image_wait_time('points', image, 900)
        i+=1

    che.show_image_wait('points', image)

    indices = che.get_indices(8, 8)

    print(len(indices))

    cells = []

    for index in indices:    
        point1 = intersection_points[index[0]]
        point2 = intersection_points[index[1]]
        point3 = intersection_points[index[2]]
        point4 = intersection_points[index[3]]

        cells.append([point1, point2, point4, point3])
        po1 = np.array([point1, point2, point4, point3], np.int32)
        
        if po1.shape[0] == 4:            
            po1 = po1.reshape((-1, 1, 2))
            image = che.draw_polygon(image, po1, (0, 255, 0), 2)
        else:
            print('Error')
            print(po1.shape[0])
            print(po1)

        #che.show_image_wait_time('points', image, 900)

    che.show_image_wait('points', image)
    
    #64 54
    point_prueba = (200, 180)
    che.draw_point(image, point_prueba, (0, 0, 255), 5)
    print(che.point_into_cell(point_prueba, cells[18]))
    che.show_image_wait('points', image)
    
    print(image.shape)
    chess_cells = che.coordinate_cells(cells, True)
    
    for chess_cell in chess_cells:
        print(chess_cell, chess_cells[chess_cell])
    
    che.destroy_windows()
    
    #dranw cells by chess coordinates
    coodinate = 'h1'
    cell = chess_cells[coodinate]
    po1 = np.array([cell[0], cell[1], cell[3], cell[2]], np.int32)
    po1 = po1.reshape((-1, 1, 2))
    image = che.draw_polygon(image, po1, (220, 120 , 10), 2)

    che.show_image_wait('points', image)

if __name__ == '__main__':
    main()  