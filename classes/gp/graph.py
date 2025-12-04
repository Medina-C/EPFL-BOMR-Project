from typing import List

import networkx as nx
from shapely.geometry import LineString

from .utils import Point, dist, same


class VisibilityGraph:
    def __init__(self, vertices: List[Point], polys):
        # Store all graph vertices and obstacles (polygons)
        self.vertices = list(vertices)
        self.polys = polys
        # Pre-build the visibility graph between all vertices
        self.base = self._build()

    def _build(self):
        # Build base visibility graph (without start/goal)
        G = nx.Graph()
        G.add_nodes_from(self.vertices)
        n = len(self.vertices)

        # Check every pair of vertices for line-of-sight visibility
        for i in range(n):
            for j in range(i + 1, n):
                a = self.vertices[i]
                b = self.vertices[j]
                # Add an edge if segment ab is not obstructed
                if self._visible(a, b):
                    G.add_edge(a, b, weight=dist(a, b))
        return G

    def with_terminals(self, start: Point, goal: Point) -> nx.Graph:
        # Create an extended visibility graph including start and goal
        G = self.base.copy()
        G.add_node(start)
        G.add_node(goal)

        # Connect start/goal to all existing visibility graph vertices
        for v in self.vertices:
            if self._visible(start, v):
                G.add_edge(start, v, weight=dist(start, v))
            if self._visible(goal, v):
                G.add_edge(goal, v, weight=dist(goal, v))

        # Direct edge start -> goal if visible
        if self._visible(start, goal):
            G.add_edge(start, goal, weight=dist(start, goal))

        return G

    def _visible(self, a: Point, b: Point) -> bool:
        # Check whether the straight line segment between a, b is obstacle-free
        seg = LineString([a, b])

        for poly in self.polys:
            # If the segment lies fully inside a polygon or crosses it -> blocked
            if seg.within(poly) or seg.crosses(poly):
                return False

            # Intersection with polygon boundary
            inter = seg.intersection(poly.boundary)

            if inter.is_empty:
                # No boundary intersection -> OK
                continue

            # Intersection is exactly one point
            if inter.geom_type == "Point":
                p = inter.coords[0]
                # Allow touching only at endpoints, otherwise blocked
                if not (same(p, a) or same(p, b)):
                    return False

            # Multiple point intersections
            elif inter.geom_type == "MultiPoint":
                # All intersection points must be exactly endpoints
                for p in inter.geoms:
                    c = p.coords[0]
                    if not (same(c, a) or same(c, b)):
                        return False

            # Allow running along the boundary
            elif inter.geom_type in ("LineString", "MultiLineString"):
                continue

            else:
                # Any unusual geometry -> treat as blocked for safety
                return False

        return True
