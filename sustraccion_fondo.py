import cv2
i=0
# Capturar video
cap = cv2.VideoCapture(0)

# Inicializar GMM para modelado de fondo
fgbg = cv2.createBackgroundSubtractorMOG2()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Aplicar el modelo GMM
    fgmask = fgbg.apply(frame)

    # Mostrar el resultado
    cv2.imshow("Foreground Mask", fgmask)
    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
