
import heapq

from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
from scipy.spatial import KDTree
from anomaly_detect.utils import BBox


class Component(object):
    def __init__(self, cat: str, bbox: BBox):
        self.cat: str = cat
        self.bbox: BBox = bbox
        self.children: List[Component] = list()
        self.cat2children: Dict[str, Component] = dict()
        self.parent: Component = None

    def add_child(self, child: 'Component'):
        child.parent = self
        self.children.append(child)
        if child.cat not in self.cat2children:
            self.cat2children[child.cat] = [child]
        else:
            self.cat2children[child.cat].append(child)

    def get_children(self, cat: str = None):
        if cat is None:
            return self.children
        return self.cat2children.get(cat, [])

    def __str__(self) -> str:
        return f'Component({self.cat}, {self.bbox})'

    def __repr__(self) -> str:
        return self.__str__()

    def __lt__(self, other: 'Component'):
        return self.bbox.area < other.bbox.area

    def __le__(self, other: 'Component'):
        return self.bbox.area <= other.bbox.area



class PositionStructure(object):
    def __init__(self, annos: Dict[str, List[BBox]], overlap_ratio=0.9):
        self.annos = annos
        self.overlap_ratio = overlap_ratio
        self.components: List[Component] = list()
        self._components_heap: List[List[float, Component]] = list()
        self.cat2components: Dict[str, Component] = defaultdict(list)
        self.idx2component: Dict[int, Component] = {}
        self.kdtree: KDTree = None
        # xmin, ymin, xmax, ymax
        n_bboxes = sum([len(bboxes) for bboxes in annos.values()])
        self.position: np.ndarray = np.zeros(
            [n_bboxes, 4], dtype=np.float32
        )
        self.build_kdtree_and_position()
        self.build_structure()
        # self.finetune_spike()

    def finetune_spike(self):
        # if find bolt_in, 2 components and 1 opening, then there is a spike
        components = self.get_components_by_cat('component')
        for component in components:
            parent = component.parent

    def get_components_by_ball(self, position: np.ndarray, r, q=2):
        indices = self.kdtree.query_ball_point(position, r, q)
        return [self.idx2component[idx] for idx in indices]

    def get_components_by_bbox(self, bbox: BBox):
        indices = np.where(np.all([
            self.position[:, 0] >= bbox.xmin - 0.01,
            self.position[:, 1] >= bbox.ymin - 0.01,
            self.position[:, 2] <= bbox.xmax + 0.01,
            self.position[:, 3] <= bbox.ymax + 0.01,
        ], axis=0))[0]
        return [self.idx2component[idx] for idx in indices]

    def get_components_by_cat(self, cat):
        return self.cat2components.get(cat, [])

    def build_structure(self):
        added = set()
        while self._components_heap:
            comp = heapq.heappop(self._components_heap)
            possible_children = self.get_components_by_ball(
                comp.bbox.center,
                max(comp.bbox.width, comp.bbox.height),
                q = 1
            )
            for possible_child in possible_children:
                if possible_child is comp or possible_child in added:
                    continue
                # NOTE: overlap > 90%
                if comp.bbox.contains(possible_child.bbox, self.overlap_ratio):
                    comp.add_child(possible_child)
                    added.add(possible_child)

    def build_kdtree_and_position(self):
        positions = []
        cnt = 0
        for cat, bboxes in self.annos.items():
            for bbox in bboxes:
                comp = Component(cat, bbox)
                self.components.append(comp)
                heapq.heappush(self._components_heap, comp)
                self.idx2component[cnt] = comp
                self.cat2components[cat].append(comp)
                positions.append(bbox.center)
                self.position[cnt] = np.array([
                    bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax
                ])
                cnt += 1
        positions = np.array(positions)
        self.kdtree = KDTree(positions)

    def __str__(self) -> str:
        return f'PositionStructure({self.components})'

    def __repr__(self) -> str:
        return self.__str__()
