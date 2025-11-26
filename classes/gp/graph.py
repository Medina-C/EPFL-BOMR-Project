from typing import List

import networkx as nx
from shapely.geometry import LineString

from .utils import Point, dist, same


class VisibilityGraph:
    def __init__(self, vertices: List[Point], polys):
        self.vertices = list(vertices)
        self.polys = polys
        self.base = self._build()

    def _build(self):
        G = nx.Graph()
        G.add_nodes_from(self.vertices)
        n = len(self.vertices)
        for i in range(n):
            for j in range(i + 1, n):
                a = self.vertices[i]
                b = self.vertices[j]
                if self._visible(a, b):
                    G.add_edge(a, b, weight=dist(a, b))
        return G

    def with_terminals(self, start: Point, goal: Point) -> nx.Graph:
        G = self.base.copy()
        G.add_node(start)
        G.add_node(goal)
        for v in self.vertices:
            if self._visible(start, v):
                G.add_edge(start, v, weight=dist(start, v))
            if self._visible(goal, v):
                G.add_edge(goal, v, weight=dist(goal, v))
        if self._visible(start, goal):
            G.add_edge(start, goal, weight=dist(start, goal))
        return G

    def _visible(self, a: Point, b: Point) -> bool:
        seg = LineString([a, b])
        for poly in self.polys:
            # Block segments strictly inside the polygon or crossing it
            if seg.within(poly) or seg.crosses(poly):
                return False

            inter = seg.intersection(poly.boundary)
            if inter.is_empty:
                continue

            # Single vertex intersection
            if inter.geom_type == "Point":
                p = inter.coords[0]
                if not (same(p, a) or same(p, b)):
                    return False

            # Multiple vertex intersections
            elif inter.geom_type == "MultiPoint":
                for p in inter.geoms:
                    c = p.coords[0]
                    if not (same(c, a) or same(c, b)):
                        return False

            # Overlap with boundary (LineString / MultiLineString) – allowed
            elif inter.geom_type in ("LineString", "MultiLineString"):
                # Segment runs along the inflated obstacle boundary: this is ok.
                continue

            else:
                # Any other weird intersection – be conservative and block
                return False

        return True
