import cv2

# Variable global para almacenar las coordenadas de inicio y fin del rectángulo
roi_coords = {"x1": 0, "y1": 0, "x2": 0, "y2": 0}
drawing = False  # Indica si el usuario está dibujando el rectángulo

# Función para manejar eventos del mouse
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


# Inicializar la captura de video desde la webcam (2 es el índice de la cámara en este caso)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("Error: No se puede abrir la cámara")
    exit()

# Crear una ventana y asignarle el callback del mouse
cv2.namedWindow("Webcam")
cv2.setMouseCallback("Webcam", draw_rectangle)

while True:
    # Capturar frame por frame
    ret, frame = cap.read()

    if not ret:
        print("Error: No se puede recibir frame (stream end?). Saliendo ...")
        break

    # Si el usuario está dibujando, muestra el rectángulo en el frame
    if drawing or (roi_coords["x1"] != roi_coords["x2"] and roi_coords["y1"] != roi_coords["y2"]):
        x1, y1, x2, y2 = roi_coords["x1"], roi_coords["y1"], roi_coords["x2"], roi_coords["y2"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Mostrar el frame capturado
    cv2.imshow('Webcam', frame)

    # Salir del bucle si se presiona la tecla 'q'
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('s') and not drawing:
        # Si se presiona 's', guardar el área seleccionada
        x1, y1, x2, y2 = roi_coords["x1"], roi_coords["y1"], roi_coords["x2"], roi_coords["y2"]
        if x1 != x2 and y1 != y2:  # Asegurarse de que el ROI no sea un rectángulo vacío
            roi = frame[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]
            cv2.imshow("ROI", roi)
            cv2.imwrite("roi.png", roi)
            print("ROI guardado como 'roi.png'")

# Liberar la captura de video y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
