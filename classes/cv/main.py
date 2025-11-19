from vision import Vision
from classes.gp.planner import GlobalPlanner
from classes.gp.plotter import Plotter

RADIUS = 2.0
SPACING = 5.0

# data = Vision.something

if __name__ == "__main__":
    planner = GlobalPlanner(data, radius=RADIUS, spacing=SPACING)
    plotter = Plotter(planner)

    segments = planner.plan()
    plotter.plain()
    plotter.visibility()
    plotter.path()
