import cv2

cap = cv2.VideoCapture(0)
ret, prev_frame = cap.read()
prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_frame, gray_frame)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    cv2.imshow("Movimiento", thresh)
    prev_frame = gray_frame

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
