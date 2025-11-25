import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_single():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)

    marker_image = cv2.aruco.generateImageMarker(aruco_dict, 0, 1000)
    cv2.imwrite('aruco0.png', marker_image)

generate_single()
