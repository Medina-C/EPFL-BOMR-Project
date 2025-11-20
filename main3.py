import numpy as np

from classes.gp import GlobalPlanner, Plotter
from classes.cv import Vision

RADIUS = 2.0
SPACING = 5.0

# Vision code either here
# Create data with vision...
# have to do something like:
# vision = Vision(args)
# data = vision.<some method>
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


if __name__ == "__main__":
    # Or Vision code here
    # Create data with vision...
    # have to do something like:
    # vision = Vision(args)
    # data = vision.<some method>
    planner = GlobalPlanner(data, radius=RADIUS, spacing=SPACING)
    plotter = Plotter(planner)

    segments = planner.plan()
    plotter.plain()
    plotter.visibility()
    plotter.path()
