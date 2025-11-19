import numpy as np
import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, planner):
        self.planner = planner

    def _setup_axes(self, ax, title):
        ax.set_xlim(0, self.planner.width)
        ax.set_ylim(self.planner.height, 0)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", "box")
        ax.set_title(title)
        ax.minorticks_on()
        ax.grid(which="both", linestyle=":", linewidth=0.5)
        ax.set_axisbelow(True)

    def _draw_obstacles(self, ax):
        for poly_pts in self.planner.obstacles:
            poly = np.asarray(poly_pts)
            xs = np.append(poly[:, 0], poly[0, 0])
            ys = np.append(poly[:, 1], poly[0, 1])
            ax.fill(
                xs,
                ys,
                alpha=0.9,
                facecolor="black",
                edgecolor="black",
                zorder=3
            )

    def _draw_inflated_and_vertices(self, ax):
        for poly in self.planner.polys:
            x, y = poly.exterior.xy
            ax.fill(
                x,
                y,
                alpha=0.25,
                facecolor="black",
                edgecolor="none",
                zorder=2
            )
        if self.planner.verts:
            xs, ys = zip(*self.planner.verts)
            ax.scatter(xs, ys, s=10, color="tab:red", alpha=0.5, zorder=4)

    def _draw_start_goal(self, ax):
        s = self.planner.start
        g = self.planner.goal
        ax.scatter(*s, s=10, color="tab:blue", zorder=5)
        ax.text(*s, "S", ha="right", va="bottom", zorder=5)
        ax.scatter(*g, s=10, color="tab:green", zorder=5)
        ax.text(*g, "G", ha="left", va="bottom", zorder=5)

    def plain(self):
        fig, ax = plt.subplots()
        self._setup_axes(ax, "Plain course")
        self._draw_obstacles(ax)
        self._draw_start_goal(ax)
        plt.show()

    def visibility(self):
        fig, ax = plt.subplots()
        self._setup_axes(ax, "Visibility Graph")
        self._draw_obstacles(ax)
        self._draw_inflated_and_vertices(ax)
        self._draw_start_goal(ax)
        G = self.planner.G
        if G is not None:
            for u, v in G.edges():
                xs, ys = zip(u, v)
                ax.plot(
                    xs,
                    ys,
                    linestyle="--",
                    linewidth=0.7,
                    color="tab:red",
                    alpha=0.5,
                    zorder=3
                )
        plt.show()

    def path(self):
        fig, ax = plt.subplots()
        self._setup_axes(ax, "A* Path")
        self._draw_obstacles(ax)
        self._draw_inflated_and_vertices(ax)
        self._draw_start_goal(ax)
        G = self.planner.G
        if G is not None:
            for u, v in G.edges():
                xs, ys = zip(u, v)
                ax.plot(
                    xs,
                    ys,
                    linestyle="--",
                    linewidth=0.5,
                    color="tab:red",
                    alpha=0.5,
                    zorder=2.5
                )
        segments = self.planner.segments
        if segments:
            xs = [p[0] for p in segments]
            ys = [p[1] for p in segments]
            ax.plot(xs, ys, linewidth=1.4, color="tab:blue", zorder=6)
            ax.scatter(xs, ys, s=5, color="tab:blue", zorder=6)
        plt.show()
