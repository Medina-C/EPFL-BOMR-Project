import cv2
import numpy as np

dpi = 600
mm_to_px = dpi / 25.4  # mm -> pixels

page_w_mm, page_h_mm = 210, 297  # A4 dimensions in mm
page_w, page_h = int(page_w_mm * mm_to_px), int(page_h_mm * mm_to_px)

tag_size_mm = 80
tag_size_px = int(tag_size_mm * mm_to_px)

# aruco dictionary and ID
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
tag_id = 5

page = 255 * np.ones((page_h, page_w), dtype=np.uint8)

tag = cv2.aruco.generateImageMarker(dictionary, tag_id, tag_size_px)

margin_mm = 15
margin_px = int(margin_mm * mm_to_px)
#x, y = margin_px, margin_px     #top left
#x, y = margin_px, page_h - tag_size_px - margin_px  # bottom left 
x, y = page_w - tag_size_px - margin_px, page_h - tag_size_px - margin_px  # bottom right
#x, y = page_w - tag_size_px - margin_px, margin_px  # top right 

# paste tag onto the page
page[y:y + tag_size_px, x:x + tag_size_px] = tag

cv2.imwrite("aruco_5target.png", page)

