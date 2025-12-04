import cv2
import numpy as np


# this file has been largely adapted from online forums and the OpenCV documentation

#create charuco board with 4x4cm chessboard square size
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
charuco_board = cv2.aruco.CharucoBoard((5, 7), 0.04, 0.02, dictionary)
board_img = charuco_board.generateImage((1000, 1400))

#create printable pdf of board
cv2.imwrite('charuco_board.png', board_img)

#collect images from camera
cap = cv2.VideoCapture(1)
all_corners = []
all_ids = []
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

print('SPACE to capture, ESC to finish')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    charuco_corners = None
    charuco_ids = None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)

    if ids is not None and len(ids) > 0:
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=charuco_board
        )

        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        if charuco_corners is not None and len(charuco_corners) > 4:
            cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)

    cv2.imshow("Calibration", frame)
    key = cv2.waitKey(1)

    if key == 32:  # SPACE
        if (
            charuco_corners is not None
            and charuco_ids is not None
            and len(charuco_corners) > 4
            and len(charuco_corners) == len(charuco_ids)
        ):
            all_corners.append(charuco_corners)
            all_ids.append(charuco_ids)
            print("Captured frame:", len(all_corners))
        else:
            print(" Skipped frame — insufficient Charuco detections")

    elif key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()


#run calibration
if len(all_corners) > 5:
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_corners,
        charucoIds=all_ids,
        board=charuco_board,
        imageSize=gray.shape[::-1],
        cameraMatrix=None,
        distCoeffs=None
    )
    print('calibration done')
    print(f'camera matrix:\n {camera_matrix}')
    print(f'distortion coefficients: \n {dist_coeffs}')
else:
    print('not enough valid frames for calibration')

print(f'calibration error={ret}')

#save parameters
np.savez('camera_calibration_param.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
