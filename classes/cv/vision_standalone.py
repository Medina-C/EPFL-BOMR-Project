import numpy as np
import cv2 as cv
import math
import time
#import constants

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
        self.camera_calib = np.load('classes/cv/vision_auxiliary/camera_calib_param.npz')
        self.frame_raw = None
        self.frame_undist = None
        self.frame_flat = None
        self.corners = None
        self.ids = None
        self.H = None
        self.scale = None
        self.obstacles = None
        self.frame_disp = None
        self.thymio_pose = None
        self.target_pose = None

    def start_video_capture(self, cam_index):
        cap = cv.VideoCapture(
            cam_index, 
            # cv.CAP_DSHOW
        )
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

        # Use consistent ordering: TL, TR, BR, BL
        src_points = np.array([
            id_to_corner[0][0][0],  # TL
            id_to_corner[1][0][3],  # BL
            id_to_corner[2][0][1],  # TR
            id_to_corner[3][0][2]   # BR
        ], dtype=np.float32)

        # Output (flattened) map size (units: mm but used as px coords for warp)
        """ w = 4*A4_W
        h = 3*A4_H """
        w = 1260
        h = 890
        dst_points = np.array([
            [0, 0],
            [0, h],
            [w, 0],
            [w, h]
        ], dtype=np.float32)

        self.H, _ = cv.findHomography(src_points, dst_points)
        # scale: mm per "pixel" in the flat output (since dst points are in mm units)
        #self.scale = IRL_MAP_DIST_W / w   # mm per dst-unit
        self.scale = 1260 / w
        return True
        
    def compute_flat_pic(self):
        if self.H is None or self.frame_undist is None:
            print("[WARN] Cannot flatten picture: H not computed yet.")
            return None

        """ w = 4*A4_W
        h = 3*A4_H """
        w = 1260
        h = 890
        self.frame_flat = cv.warpPerspective(self.frame_undist, self.H, (w, h))
        return True

    def get_tag_pose(self, id):
        if self.ids is None:
            print(f'[VISION] no tags detected currently')
            return 
        ids_flat = self.ids.flatten()
        if id in ids_flat:
            idx = np.where(ids_flat == id)[0][0]
            target_corners = self.corners[idx][0]
        else: 
            print(f'[VISION] tag id= {id} not detected')
            return

        pts = target_corners.reshape(-1, 1, 2)
        target_center_px = None
        target_center_mm = None
        theta = None
        # compute flat coordinates if homography exists
        if self.H is not None:
            target_corners_flat = cv.perspectiveTransform(pts, self.H)
            target_center_px = target_corners_flat.mean(axis=0)  # in flat units
            target_center_mm = target_center_px * self.scale     # mm units

            # also compute undistorted pixel location for drawing by inverting H
            try:
                H_inv = np.linalg.inv(self.H)
                pts_flat = target_center_px.reshape(-1,1,2).astype(np.float32)
                pts_undist = cv.perspectiveTransform(pts_flat, H_inv)
                cx, cy = pts_undist.reshape(-1)
                tag_undist_px = np.array([int(round(float(cx))), int(round(float(cy)))])
            except Exception:
                tag_undist_px = None

        # 2D orientation in undistorted image (from corners)
        v = target_corners[0] - target_corners[3]   #<--------------- if angle not correct, use pts_undist.reshape(4,2)[1] - pts_undist.reshape(4,2)[0]
        theta =  math.atan2(v[1], v[0]) #* (180/math.pi)
        if id == 4:     #replace with use of constants file
            self.thymio_pose = {'pose_mm': target_center_mm,
                                'pose_undist_px': tag_undist_px,
                                'theta': theta}
        elif id == 5:   #same
            self.target_pose = {'pose_mm': target_center_mm,
                                'pose_undist_px': tag_undist_px,
                                'theta': theta}
    

    def get_tag_velocity(self, id):
        self.get_tag_pose(id)
        pose_map = {4: self.thymio_pose, 5: self.target_pose}
        pose = pose_map.get(id)
        
        if pose is None or pose.get("pose_mm") is None:
            return None
        
        now = time.time()
        pos = np.array(pose["pose_mm"], dtype=float).reshape(-1)
        theta = float(pose["theta"])  # direct accessor, always exists if pose
        
        if not hasattr(self, "_last_tag_state"):
            self._last_tag_state = {
                4: None
            }
        
        prev = self._last_tag_state[id]
        
        self._last_tag_state[id] = {"t": now, "pos": pos, "theta": theta}
        
        if prev is None:
            return {
                "v_mm_s": np.array([0.0, 0.0]),
                "speed_mm_s": 0.0,
                "omega_rad_s": 0.0,
                "dt": None,
            }
        dt = now - prev["t"]
        if dt <= 0:
            return {
                "v_mm_s": np.array([0.0, 0.0]),
                "speed_mm_s": 0.0,
                "omega_rad_s": 0.0,
                "dt": dt,
            }
        dpos = (pos - prev["pos"]) / dt
        dth = (theta - prev["theta"]) / dt
        
        return {
            "v_mms_s": dpos,
            "speed_mm_s": float(np.linalg.norm(dpos)),
            "omega_rad_s": dth,
            "dt": dt
        }

    def find_obstacle_vertices(self, img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (5, 5), 0)

        # Crop borders
        crop_margin = 1
        cropped = gray[crop_margin:-crop_margin, crop_margin:-crop_margin]

        # Adaptive thresholding for robustness
        threshold = cv.adaptiveThreshold(
            cropped, 255,
            cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY_INV,
            51,
            5
        )

        # Morphology to remove small specks
        kernel = np.ones((3, 3), np.uint8)
        threshold = cv.morphologyEx(threshold, cv.MORPH_OPEN, kernel, iterations=1)

        cv.imwrite("classes/cv/results/debug_threshold.png", threshold)

        contours, _ = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        obstacles = []
        obstacle_info_gv = []
        h_c, w_c = cropped.shape
        obstacle_info_gv.append(np.array([w_c, h_c]))

        # obstacle must be at least 85x85 mm
        min_obstacle_mm = 100
        if self.scale is None or self.scale == 0:
            min_area_px = 0
        else:
            min_area_px = (min_obstacle_mm / self.scale) ** 2

        for cont in contours:
            area = cv.contourArea(cont)
            if area < min_area_px:
                continue

            perimeter = cv.arcLength(cont, True)
            epsilon = 0.015 * perimeter
            approx = cv.approxPolyDP(cont, epsilon, True)

            if not cv.isContourConvex(approx):
                approx = cv.convexHull(approx)

            approx_px = approx.reshape(-1, 2)
            approx_px_full = approx_px + np.array([crop_margin, crop_margin])

            approx_mm = approx_px_full * self.scale if self.scale is not None else approx_px_full

            obstacles.append({
                "vertices_px": approx_px_full,
                "vertices_mm": approx_mm,
                "area_px": area,
                "num_vertices": len(approx_px),
            })
            obstacle_info_gv.append(np.array(approx_mm))
        self.obstacles = obstacle_info_gv.copy()

        debug = img.copy()
        for obs in obstacles:
            cv.polylines(debug, [obs["vertices_px"].astype(int)], True, (0, 0, 255), 2)
        cv.imwrite("classes/cv/results/debug_obstacles.png", debug)
        return obstacles

    def update_vision(self, save_frames, find_flat, find_obstacles):
        #print('[VISION] running update_vision')
        ret, frame = self.cap.read()
        if not ret:
            print('[VISION] cannot read camera frame')
            return None

        self.frame_raw = frame
        self.frame_undist = self.remove_distortion(frame)
        self.corners, self.ids = self.get_tag_positions(self.frame_undist)

        frame_show = self.frame_undist.copy()
        if self.ids is not None:
            cv.aruco.drawDetectedMarkers(frame_show, self.corners, self.ids)
        self.frame_disp = cv.resize(frame_show, (DISP_FRAME_W, DISP_FRAME_H))

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

        # draw overlays on frame_disp if available
        self.update_disp_frame()
  

    def return_dummy(self):
        data = [
        [260.0, 200.0],
        np.array([[60, 40], [90, 40], [90, 80], [60, 80]]),
        np.array([[130, 40], [200, 40], [200, 70], [130, 70]]),
        np.array([[120, 120], [160, 120], [160, 170], [120, 170]]),
        np.array([[40, 130], [70, 140], [65, 160], [35, 170], [25, 150]]),
        np.array([[200, 110], [230, 120], [220, 145]]),
        np.array([[180, 10], [230, 15], [225, 35], [175, 30]]),
        np.array([20, 20]),
        np.array([240, 180]),
        ]
        return data
    
    def update_disp_frame(self):
        if self.frame_disp is None:
            return
        # draw obstacles (simple visualization)
        if isinstance(self.obstacles, list):
            # obstacles stored as arrays in obstacle_info_gv; skip first entry (dims)
            for item in self.obstacles[1:]:
                #print('updating frame with obstacle drawing')
                try:
                    pts = np.array(item).astype(int)
                    if pts.ndim == 2 and pts.shape[0] >= 3:
                        # scale display coords from flat->display rough approximation
                        # we skip precise reprojection for minimal invasive change
                        x = (pts[:,0] * DISP_FRAME_W / (4*A4_W)).astype(int)
                        y = (pts[:,1] * DISP_FRAME_H / (3*A4_H)).astype(int)
                        pts_disp = np.stack([x, y], axis=1)
                        cv.polylines(self.frame_disp, [pts_disp], True, (0,255,0), 2)
                except Exception:
                    pass

        if self.thymio_pose is not None:
            cv.circle(self.frame_disp, (self.thymio_pose['pose_undist_px'][0], self.thymio_pose['pose_undist_px'][1]),
                      8, (0,0,255), -1)
        if self.target_pose is not None:
            cv.circle(self.frame_disp, (self.target_pose['pose_undist_px'][0], self.target_pose['pose_undist_px'][1]),
                      8, (0,0,255), -1)
            

        """ if self.last_tag_undist_px is not None:
            cx, cy = int(self.last_tag_undist_px[0]), int(self.last_tag_undist_px[1])
            # scale to display
            cx_disp = int(cx * DISP_FRAME_W / self.frame_undist.shape[1])
            cy_disp = int(cy * DISP_FRAME_H / self.frame_undist.shape[0])
            cv.circle(self.frame_disp, (cx_disp, cy_disp), 8, (0,0,255), -1) """


if __name__ == "__main__":
    vision = Vision(0)
    print('[VISION] press SPACE to update. press ESC to quit')
    
    while True:
        t = time.time()
        vision.update_vision(True, False, False)
        print(f"time false={time.time()-t}")
        cv.imshow("feed", vision.frame_disp)
        key = cv.waitKey(1)
        if key == 27:   #ESC
            break

        if key == 32:   #SPACE
            t = time.time()
            frame = vision.update_vision(True, True, True)
            print(f"time true={time.time()-t}")
    vision.stop_video_capture()

#pixel coordinate axis:
#(w,h) with (0,0) in TL and w increasing to the right and h increasing downwards

#black square (226,138) [mm]
