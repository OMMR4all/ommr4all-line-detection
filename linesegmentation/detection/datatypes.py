from dataclasses import dataclass
from typing import List
import numpy as np
import copy

@dataclass
class Point:
    x: float
    y: float


class Line:
    def __init__(self, line: List[Point]):
        self.line: List[Point] = line

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
