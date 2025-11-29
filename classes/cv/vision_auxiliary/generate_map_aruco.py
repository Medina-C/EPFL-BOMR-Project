import cv2
import numpy as np
import os

# -------- Settings --------
dpi = 600
mm_to_px = dpi / 25.4  # mm → pixels

page_w_mm, page_h_mm = 210, 297  # A4 dimensions in mm
page_w, page_h = int(page_w_mm * mm_to_px), int(page_h_mm * mm_to_px)

# -------- Tag size (physical: ~70 mm side) --------
tag_size_mm = 100  # chosen for robust detection from 1–2 m above
tag_size_px = int(tag_size_mm * mm_to_px)

# ArUco dictionary and ID
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
tag_id = 4

# -------- Create blank A4 page --------
page = 255 * np.ones((page_h, page_w), dtype=np.uint8)

# -------- Generate the tag --------
tag = cv2.aruco.generateImageMarker(dictionary, tag_id, tag_size_px)

# -------- Place tag in top-left corner with small margin --------
margin_mm = 15
margin_px = int(margin_mm * mm_to_px)
#x, y = margin_px, margin_px     #top left
#x, y = margin_px, page_h - tag_size_px - margin_px  # bottom left 
x, y = page_w - tag_size_px - margin_px, page_h - tag_size_px - margin_px  # bottom right
#x, y = page_w - tag_size_px - margin_px, margin_px  # top right 

# Save pixel (src) and millimeter (dst) coordinates
src_points = np.array([[x, y]])
dst_points = src_points / mm_to_px

#np.savez("single_tag_pos.npz", src_points=src_points, dst_points=dst_points)

# -------- Paste the tag onto the page --------
page[y:y + tag_size_px, x:x + tag_size_px] = tag

# -------- Save the image --------
cv2.imwrite("single.png", page)

print("Saved images/aruco_single_A4.png")
print("Tag physical size ≈ {:.2f} mm".format(tag_size_px / mm_to_px))
