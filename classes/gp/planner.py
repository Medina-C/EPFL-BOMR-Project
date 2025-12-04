from typing import List

import math
import numpy as np
from shapely.geometry import Polygon
import networkx as nx

from .utils import Point, dist
from .graph import VisibilityGraph


class GlobalPlanner:
    def __init__(
        self,
        data: List[np.ndarray],
        radius: float = 0.0,
        spacing: float = 5.0
    ) -> None:
        self.width, self.height = data[0]
        self.obstacles = [np.asarray(p, float) for p in data[1:-2]]
        self.start = self._clamp_point(tuple(np.asarray(data[-2], float)))
        self.goal = self._clamp_point(tuple(np.asarray(data[-1], float)))

        self.r = radius
        self.spacing = spacing

        # Build inflated polygons + visibility vertices + the visibility graph
        self.polys, self.verts = self._build_polys()
        self.vg = VisibilityGraph(self.verts, self.polys)

        self.G = None
        self.path = None
        self.segments = None

    @staticmethod
    def _angle_mean(a: float, b: float) -> float:
        return math.atan2(
            math.sin(a) + math.sin(b),
            math.cos(a) + math.cos(b),
        )

    def _clamp_point(self, p: Point) -> Point:
        # Clamp a point into [0, W]x[0, H]
        x, y = p
        x = min(max(float(x), 0.0), float(self.width))
        y = min(max(float(y), 0.0), float(self.height))
        return (x, y)

    def plan(self, start: Point = None):
        if start is not None:
            self.start = self._clamp_point(tuple(start))

        # Build visibility graph including start/goal
        self.G = self.vg.with_terminals(self.start, self.goal)

        try:
            # Run A* on the vg
            path = nx.astar_path(
                self.G,
                self.start,
                self.goal,
                heuristic=lambda a, b: dist(a, b),
                weight="weight",
            )
        except:
            # No feasible path between start and goal
            print("No path found")
            return None
        self.path = path

        segments: list[list[float]] = []
        # Map from vertex index in path to index in segments
        vertex_segment_idx: dict[int, int] = {}
        segment_thetas: list[float] = []

        # Discretize each visibility edge into evenly spaced waypoints
        for i in range(len(path) - 1):
            x0, y0 = path[i]
            x1, y1 = path[i + 1]
            dx = x1 - x0
            dy = y1 - y0
            L = math.hypot(dx, dy)
            if L == 0:
                # Skip zero-length edges
                continue
            theta = math.atan2(dy, dx)
            segment_thetas.append(theta)
            if i == 0:
                segments.append([x0, y0, theta])
                vertex_segment_idx[0] = 0

            # Go with spacing
            d = self.spacing
            while d < L:
                t = d / L
                x = x0 + t * dx
                y = y0 + t * dy
                segments.append([x, y, theta])
                d += self.spacing

            # Ensure end vertex of this edge
            segments.append([x1, y1, theta])
            vertex_segment_idx[i + 1] = len(segments) - 1
        n_vertices = len(path)
        if n_vertices >= 3:
            for j in range(1, n_vertices - 1):
                idx = vertex_segment_idx[j]
                theta_prev = segment_thetas[j - 1]
                theta_next = segment_thetas[j]
                theta_avg = self._angle_mean(theta_prev, theta_next)
                segments[idx][2] = theta_avg

        # Store and return the final waypoint list
        self.segments = segments
        return segments

    def _build_polys(self):
        polys = []
        verts = []
        for obs in self.obstacles:
            # Construct polygon from obstacle vertices
            poly = Polygon(obs)

            # Inflate polygon by robot radius
            if self.r > 0:
                poly = poly.buffer(self.r, join_style=2)

            polys.append(poly)
            for x, y in list(poly.exterior.coords)[:-1]:
                # Clamp vertices to map bounds before adding them
                x = min(max(float(x), 0.0), float(self.width))
                y = min(max(float(y), 0.0), float(self.height))
                verts.append((x, y))

        return polys, verts
