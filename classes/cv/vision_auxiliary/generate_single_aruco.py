import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_single():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)

    marker_image = cv2.aruco.generateImageMarker(aruco_dict, 4, 200)
    cv2.imwrite('images/aruco4.png', marker_image)

generate_single()
