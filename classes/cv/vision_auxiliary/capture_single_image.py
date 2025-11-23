import cv2

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
if not cap.isOpened():
    print("Could not open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("img1", frame)

    key = cv2.waitKey(1)
    if key == ord('y'):
        cv2.imwrite("im1.png", frame)
        print("Image saved as images/c1.png")
        break
    elif key == ord('q'):
        print("Quit without saving")
        break

cap.release()
cv2.destroyAllWindows()


