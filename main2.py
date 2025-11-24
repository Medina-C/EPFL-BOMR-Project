from classes.cv.vision_standalone import Vision
from classes.gp.planner import GlobalPlanner
from classes.gp.plotter import Plotter
import cv2 as cv
import constants
import numpy as np

RADIUS = 2.0
SPACING = 5.0


if __name__ == "__main__":
    vision = Vision(1)
    data = []
    while True:
        vision.update_vision(True, False, False)    

        if vision.frame_disp is not None:
            cv.imshow('frame', vision.frame_disp)
        else:
            blank = 255 * np.ones((480, 640, 3), dtype=np.uint8)
            cv.imshow('frame', blank)

        key = cv.waitKey(1) & 0xFF
        if key == 27:  # ESC -> exit
            break
        if key == ord('h'):
            vision.update_vision(True, True, True)
            vision.get_tag_pose(4)

            if vision.obstacles is not None:
                data = vision.obstacles.copy()
            if vision.thymio_pose is not None:
                data.append(vision.thymio_pose['pose_mm'].flatten())
            if vision.target_pose is not None:
                data.append(vision.target_pose['pose_mm'].flatten())
                print(f'data+thy+tar= {data}')
                data.append(np.array([100, 100]))


        if key == ord('p'):
            data.append(np.array([100, 100]))
            planner = GlobalPlanner(data, radius=RADIUS, spacing=SPACING)
            plotter = Plotter(planner)

            segments = planner.plan()
            plotter.plain()
            plotter.visibility()
            plotter.path()
