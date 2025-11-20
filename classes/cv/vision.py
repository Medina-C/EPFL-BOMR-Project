import numpy as np
import cv2 as cv
import math
import time

A4_W = 210  #in mm
A4_H = 297  #in mm
IRL_MAP_DIST_W = 4*A4_W - 28    #in mm

DISP_FRAME_W = 960
DISP_FRAME_H = 540


class Vision:
    def __init__(self, cam_index):
        self.cap = self.start_video_capture(cam_index)
        self.aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_50)
        params = cv.aruco.DetectorParameters()
        self.detector = cv.aruco.ArucoDetector(self.aruco_dict, params)
        self.camera_calib = np.load('classes\\cv\\vision_auxiliary\\camera_calib_param.npz')
        self.frame_raw = None
        self.frame_undist = None
        self.frame_flat = None
        self.corners = None
        self.ids = None
        self.H = None
        self.scale = None
        self.obstacles = None
        self.tags = None
        self.frame = None

    def start_video_capture(self, cam_index):
        cap = cv.VideoCapture(cam_index, cv.CAP_DSHOW)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
        return cap

    def stop_video_capture(self):
        self.cap.release()
        cv.destroyAllWindows()

    def get_tag_positions(self, frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        return corners, ids

    def remove_distortion(self, frame):
        h, w = frame.shape[:2]
        new_camera_matrix, _ = cv.getOptimalNewCameraMatrix(self.camera_calib['camera_matrix'],
                                self.camera_calib['dist_coeffs'],(w, h), 1, (w, h))

        undist = cv.undistort(frame, self.camera_calib['camera_matrix'],
                            self.camera_calib['dist_coeffs'], None, new_camera_matrix)
        return undist

    def compute_Homography(self, corners, ids):
        #Compute H using only marker IDs 0,1,2,3
        if ids is None or len(corners) < 4:
            print("[ERROR] Need all 4 corner markers.")
            return None

        ids_flat = ids.flatten()

        # Filter markers with ID < 4 (corner markers)
        mask = ids_flat < 4
        ids_filtered = ids_flat[mask]
        corners_filtered = [corners[i] for i in range(len(corners)) if mask[i]]

        # Map ID -> corner points
        id_to_corner = {
            int(ids_filtered[i]): corners_filtered[i]
            for i in range(len(ids_filtered))
        }

        for nid in [0, 1, 2, 3]:
            if nid not in id_to_corner:
                print(f"[ERROR] Corner marker {nid} missing.")
                return None

        src_points = np.array([
            id_to_corner[0][0][0],  # TL
            id_to_corner[1][0][3],  # BL
            id_to_corner[2][0][1],  # TR
            id_to_corner[3][0][2]   # BR
        ], dtype=np.float32)

        # Output (flattened) map size
        w = 4*A4_W
        h = 3*A4_H
        dst_points = np.array([
            [0, 0],
            [0, h],
            [w, 0],
            [w, h]
        ], dtype=np.float32)
        self.H, _ = cv.findHomography(src_points, dst_points)
        self.scale = IRL_MAP_DIST_W / w   #px->mm      TO ADJUST depending on real life map
        return True
        
    def compute_flat_pic(self):
        if self.H is None or self.frame_undist is None:
            print("[WARN] Cannot flatten picture: H not computed yet.")
            return None

        w = 4*A4_W
        h = 3*A4_H
        self.frame_flat = cv.warpPerspective(self.frame_undist, self.H, (w, h))
        return True

    def get_tag_pose(self, id):
        ids_flat = self.ids.flatten()
        if id in ids_flat:
            idx = np.where(ids_flat == id)[0][0]     #self.ids not previously sorted so need to get idx
            target_corners = self.corners[idx][0]
        else: 
            print(f'[VISION] tag id= {id} not detected')
            return None, None

        pts = target_corners.reshape(-1, 1, 2)      #for .perspectiveTransform purposes
        if self.H is not None:
            target_corners_flat = cv.perspectiveTransform(pts, self.H)
            target_center_px = target_corners_flat.mean(axis=0)
            target_center_mm = target_center_px * self.scale 

            v = target_corners[1] - target_corners[0]
            theta =  math.atan2(v[1], v[0]) * (180/math.pi)          
            return target_center_mm, target_center_px, theta
    
    def find_obstacle_vertices(self, img):
        # ------------------------
        # 1. Preprocessing
        # ------------------------
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (5, 5), 0)

        # Crop borders
        crop_margin = 1
        cropped = gray[crop_margin:-crop_margin, crop_margin:-crop_margin]

        # Adaptive thresholding for robustness
        threshold = cv.adaptiveThreshold(
            cropped, 255,      #use cropped instead of gray if using margin removal
            cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY_INV,
            51,   # block size (odd)
            5     # C value
        )

        # Morphology to remove small specks
        kernel = np.ones((3, 3), np.uint8)
        threshold = cv.morphologyEx(threshold, cv.MORPH_OPEN, kernel, iterations=1)

        cv.imwrite("classes/cv/results/debug_threshold.png", threshold)

        # ------------------------
        # 2. Find contours
        # ------------------------
        contours, _ = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        obstacles = []

        # ------------------------
        # 3. Minimum area filter based on real size
        # ------------------------
        # obstacle must be at least 40x40 mm
        min_obstacle_mm = 85    #to ignore aruco tags
        min_area_px = (min_obstacle_mm / self.scale) ** 2  # convert mm² → px²

        for cont in contours:
            area = cv.contourArea(cont)
            if area < min_area_px:
                continue

            # ------------------------
            # 4. Polygon approximation
            # ------------------------
            perimeter = cv.arcLength(cont, True)
            epsilon = 0.015 * perimeter  # lower = more vertices
            approx = cv.approxPolyDP(cont, epsilon, True)

            if not cv.isContourConvex(approx):
                approx = cv.convexHull(approx)

            # ------------------------
            # 5. Convert coordinates to full image
            # ------------------------
            approx_px = approx.reshape(-1, 2)
            approx_px_full = approx_px + np.array([crop_margin, crop_margin])

            approx_mm = approx_px_full * self.scale  # px → mm

            obstacles.append({
                "vertices_px": approx_px_full,
                "vertices_mm": approx_mm,
                "area_px": area,
                "num_vertices": len(approx_px),
            })
            self.obstacles = obstacles.copy()

        # ------------------------
        # 6. Debug visualization
        # ------------------------
        debug = img.copy()
        for obs in obstacles:
            cv.polylines(debug, [obs["vertices_px"].astype(int)], True, (0, 0, 255), 2)
        cv.imwrite("classes/cv/results/debug_obstacles.png", debug)

        return obstacles



    def update_vision(self, save_frames, find_flat, find_obstacles):
        ret, frame = self.cap.read()
        if not ret:
            print('[VISION] cannot read camera frame')
            return

        self.frame_raw = frame
        self.frame_undist = self.remove_distortion(frame)
        self.corners, self.ids = self.get_tag_positions(self.frame_undist)

        frame_show = self.frame_undist.copy()
        if self.ids is not None:
            cv.aruco.drawDetectedMarkers(frame_show, self.corners, self.ids)
        display_frame = cv.resize(frame_show, (DISP_FRAME_W, DISP_FRAME_H))

        if save_frames:
            cv.imwrite('classes/cv/results/frame_raw.png', self.frame_raw)
            cv.imwrite('classes/cv/results/frame_undist.png', self.frame_undist)
        
        if find_flat:
            ret = self.compute_Homography(self.corners, self.ids)
            if ret:
                print('[VISION] computed homography')

            if self.H is not None:
                ret = self.compute_flat_pic()
                if self.frame_flat is not None:
                    print('[VISION] flat map created successfully')
                    if save_frames:
                        cv.imwrite('classes/cv/results/frame_flat.png', self.frame_flat)

        if find_obstacles and self.frame_flat is not None:
            self.find_obstacle_vertices(self.frame_flat)
            if self.obstacles is not None:
                print('[VISION] found obstacles')
                print(self.obstacles)
           
        """    if draw_tags and self.tags is not None:
            for tag in self.tags:
                H_inv = np.linalg.inv(vision.H)
                pts_flat = vision.target_center_px.reshape(-1,1,2)  # (1,1,2) shape for cv.perspectiveTransform
                pts_undist = cv.perspectiveTransform(pts_flat, H_inv)
                cx, cy = pts_undist.flatten()
                cx, cy = pts_undist.reshape(-1)
                cx = int(round(float(cx)))
                cy = int(round(float(cy)))
                cv.circle(frame, (cx,cy), 20, (0,0,255), -1) """
        return display_frame
  


if __name__ == "__main__":
    vision = Vision(1)
    print('[VISION] press SPACE to update. press ESC to quit')
    
    while True:
        frame = vision.update_vision(True, False, False)
        cv.imshow("feed", frame)
        key = cv.waitKey(1)
        if key == 27:   #ESC
            break

        if key == 32:   #SPACE
            #start = time.perf_counter()
            frame = vision.update_vision(True, True, True)
            #end = time.perf_counter()
            #elapsed_ms = (end - start) * 1000
            #print(f"[TIME] update_vision(true) took {elapsed_ms:.2f} ms")
    vision.stop_video_capture()

#pixel coordinate axis:
#(w,h) with (0,0) in TL and w increasing to the right and h increasing downwards

#black square (226,138) [mm]
