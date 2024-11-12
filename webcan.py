import cv2

# Inicializar la captura de video desde la webcam (0 es el índice de la cámara por defecto)
cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("Error: No se puede abrir la cámara")
    exit()

while True:
    # Capturar frame por frame
    ret, frame = cap.read()

    # Si no se pudo capturar el frame, salir del bucle
    if not ret:
        print("Error: No se puede recibir frame (stream end?). Saliendo ...")
        break

    # Mostrar el frame capturado
    cv2.imshow('Webcam', frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Liberar la captura de video y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()