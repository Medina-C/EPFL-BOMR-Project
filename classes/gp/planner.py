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

        # Clamp start/goal to map bounds
        self.start = self._clamp_point(tuple(np.asarray(data[-2], float)))
        self.goal = self._clamp_point(tuple(np.asarray(data[-1], float)))

        self.r = radius
        self.spacing = spacing

        self.polys, self.verts = self._build_polys()
        self.vg = VisibilityGraph(self.verts, self.polys)

        self.G = None
        self.path = None
        self.segments = None

    def _clamp_point(self, p: Point) -> Point:
        x, y = p
        x = min(max(float(x), 0.0), float(self.width))
        y = min(max(float(y), 0.0), float(self.height))
        return (x, y)

    def plan(self, start: Point = None) -> list:
        if start is not None:
            # Also clamp dynamic start to map
            self.start = self._clamp_point(tuple(start))

        self.G = self.vg.with_terminals(self.start, self.goal)

        path = nx.astar_path(
            self.G,
            self.start,
            self.goal,
            heuristic=lambda a, b: dist(a, b),
            weight="weight",
        )
        self.path = path

        segments = []
        for i in range(len(path) - 1):
            x0, y0 = path[i]
            x1, y1 = path[i + 1]
            dx = x1 - x0
            dy = y1 - y0
            L = math.hypot(dx, dy)
            if L == 0:
                continue
            theta = math.atan2(dy, dx)
            if i == 0:
                segments.append([x0, y0, theta])
            d = self.spacing
            while d < L:
                t = d / L
                x = x0 + t * dx
                y = y0 + t * dy
                segments.append([x, y, theta])
                d += self.spacing
            segments.append([x1, y1, theta])

        self.segments = segments
        return segments

    def _build_polys(self):
        polys = []
        verts = []
        for obs in self.obstacles:
            poly = Polygon(obs)
            if self.r > 0:
                poly = poly.buffer(self.r, join_style=2)
            polys.append(poly)

            # Clamp all polygon vertices to map bounds before using them
            for x, y in list(poly.exterior.coords)[:-1]:
                x = min(max(float(x), 0.0), float(self.width))
                y = min(max(float(y), 0.0), float(self.height))
                verts.append((x, y))

        return polys, verts
