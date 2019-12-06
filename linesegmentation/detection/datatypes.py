from dataclasses import dataclass
from typing import List
import numpy as np
import copy

@dataclass
class AABB:
    x1: float
    y1: float
    x2: float
    y2: float

    def h(self):
        return self.y2 - self.y1

    def w(self):
        return self.x2 - self.x1

    def intersects(self, other: 'AABB') -> bool:
        return not (other.x1 > self.x2 or other.x2 < self.x1 or other.y1 > self.y2 or other.y2 < self.y1)

    def expand(self, value):
        x, y = value
        self.x1 -= x
        self.y1 -= y
        self.x2 += x
        self.y2 += y

    def copy(self) -> 'AABB':
        return copy.copy(self)

@dataclass
class Point:
    x: float
    y: float


class Line:
    def __init__(self, line: List[Point]):
        self.line: List[Point] = line
        self._aabb: AABB = None

    def aabb(self) -> AABB:
        if not self._aabb:
            x1 = min([p.x for p in self.line])
            y1 = min([p.y for p in self.line])
            x2 = max([p.x for p in self.line])
            y2 = max([p.y for p in self.line])
            self._aabb = AABB(x1, y1, x2, y2)
        return self._aabb

    def get_start_point(self):
        return self.line[0]

    def get_end_point(self):
        return self.line[-1]

    def get_average_line_height(self):
        return np.mean([point.y for point in self.line])

    def get_xy(self):
        x_list = []
        y_list = []
        for point in self.line:
            x_list.append(point.x)
            y_list.append(point.y)
        return x_list, y_list

    def __len__(self):
        return len(self.line)

    def __iter__(self):
        return iter(self.line)

    def __getitem__(self, key):
        return self.line[key]

    def __setitem__(self, key, value):
        self.line[key] = value

    def __str__(self):
        return "[{0}]".format(', '.join(map(str, self.line)))

    def __repr__(self):
        return str(self)

    def __add__(self, other):
        return self.line + other.line

    def __radd__(self, other):
        return other.line + self.line

    def l_append(self, value):
        self.line = value + self.line

    def r_append(self, value):
        self.line = self.line + value

    def scale_line(self, factor: float):
        self.line = [Point(point.x * factor, point.y * factor) for point in self.line]

    def __copy__(self):
        return Line(copy.copy(self.line))

    def __delitem__(self, key):
        del self.line[key]


class System:
    def __init__(self, system: List[Line]):
        self.system: List[Line] = system

    def __len__(self):
        return len(self.system)

    def __iter__(self):
        return iter(self.system)

    def __getitem__(self, key):
        return self.system[key]

    def __setitem__(self, key, value):
        self.system[key] = value

    def __delitem__(self, key):
        del self.system[key]

    def __str__(self):
        return "[{0}]".format(', '.join(map(str, self.system)))

    def __repr__(self):
        return str(self)
