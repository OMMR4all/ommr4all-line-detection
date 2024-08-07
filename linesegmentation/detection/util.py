import peakutils
from scipy.signal import medfilt2d
import numpy as np
from itertools import tee
from scipy.interpolate import interpolate
from numpy.linalg import norm
from typing import List
from linesegmentation.detection.datatypes import Line, Point, System


def get_text_borders(image: np.ndarray, preprocess: bool = False, min_dist: int = 30, thres: float = 0.3):
    med = image.copy()
    if preprocess:
        med = medfilt2d(image, 9)
    histogram = np.sum(med == 255, axis=1)
    text_borders = peakutils.indexes(histogram, thres=thres, min_dist=min_dist)
    return text_borders


def vertical_runs(img: np.array) -> [int, int]:
    img = np.transpose(img)
    h = img.shape[0]
    w = img.shape[1]
    transitions = np.transpose(np.nonzero(np.diff(img)))
    white_runs = [0] * (w + 1)
    black_runs = [0] * (w + 1)
    a, b = tee(transitions)
    next(b, [])
    for f, g in zip(a, b):
        if f[0] != g[0]:
            continue
        tlen = g[1] - f[1]
        if img[f[0], f[1] + 1] == 1:
            white_runs[tlen] += 1
        else:
            black_runs[tlen] += 1

    for y in range(h):
        x = 1
        col = img[y, 0]
        while x < w and img[y, x] == col:
            x += 1
        if col == 1:
            white_runs[x] += 1
        else:
            black_runs[x] += 1

        x = w - 2
        col = img[y, w - 1]
        while x >= 0 and img[y, x] == col:
            x -= 1
        if col == 1:
            white_runs[w - 1 - x] += 1
        else:
            black_runs[w - 1 - x] += 1

    black_r = np.argmax(black_runs) + 1
    # on pages with a lot of text the staffspaceheigth can be falsified.
    # --> skip the first elements of the array, we expect the staff lines distance to be at least twice the line height
    white_r = np.argmax(white_runs[black_r * 3:]) + 1 + black_r * 3
    return white_r, black_r


def angle_difference_of_points(p1: Point, p2: Point) -> float:
    v1 = np.array([p1.x, p1.y])
    v2 = np.array([p2.x, p2.y])

    angle_difference = np.arccos((v1 @ v2) / (norm(v1) * norm(v2)))

    return angle_difference


def simplify_anchor_points(line: Line, max_distance: int = 25, min_distance: int = 10,
                           min_degree_to_keep_points: float = 0.2):
    new_line = []

    def distance(p1: Point, p2: Point):
        return p2.x - p1.x
    prev_point: Point = None
    for point_ind, point in enumerate(line):
        if prev_point is not None:
            point_distance = distance(prev_point, point)
        else:
            point_distance = min_distance + 1
        if prev_point is not None and point_distance > max_distance:
            new_line.append(Point(prev_point.x + (point.x - prev_point.x) / 2, point.y)) # [point[0], prev_point[1] + (point[1] - prev_point[1])/2])
        if prev_point is not None and point_distance < min_distance:
            if angle_difference_of_points(prev_point, point) > min_degree_to_keep_points or point_ind == len(line) - 1:
                new_line.append(point)
        else:
            new_line.append(point)
        prev_point = point
    return Line(new_line)


def best_line_fit(img: np.array, line: Line, line_thickness: int = 3, max_iterations: int = 30,
                  scale: float = 1.0, skip_startend_points: bool = False) -> Line:
    current_blackness = get_blackness_of_line(line, img)
    best_line = line.__copy__()

    change = True
    iterations = 0

    while change:
        if iterations > max_iterations:
            break
        change = False
        for point_ind, point in enumerate(best_line):
            if skip_startend_points:
                if point_ind == 0 or point_ind == len(best_line):
                    continue
            y, x = point.y, point.x
            scaled_line_thickness = line_thickness * np.ceil(scale).astype(int)
            for i in range(1, scaled_line_thickness + 1):
                if y + i < line[point_ind].y + scaled_line_thickness:
                    test_line = best_line.__copy__()
                    test_line[point_ind] = Point(x, y + i)
                    blackness = get_blackness_of_line(test_line, img)

                    if blackness < current_blackness:
                        change = True
                        current_blackness = blackness
                        best_line[point_ind] = Point(x, y + i)
                if y - i > line[point_ind].y - scaled_line_thickness:

                    test_line[point_ind] = Point(x, y - i)
                    blackness = get_blackness_of_line(test_line, img)

                    if blackness < current_blackness:
                        change = True
                        current_blackness = blackness
                        best_line[point_ind] = Point(x, y - i)

        iterations += 1
    return best_line


def get_blackness_of_line(line: Line, image: np.ndarray) -> int:
    x_list, y_list = line.get_xy()
    func = interpolate.interp1d(x_list, y_list)
    x_start, x_end = int(x_list[0]), int(x_list[-1])
    x_list_new = np.arange(x_start, x_end-1)
    y_new = func(x_list_new)
    y_new[y_new > image.shape[0] - 1] = image.shape[0] - 1
    y_new_int = np.floor(y_new + 0.5).astype(int)
    indexes = (np.array(y_new_int), np.array(x_list_new))

    blackness = np.mean(image[indexes])
    return blackness


# Used for testing
def get_blackness_of_line_distribution(line: List[List[int]], image: np.ndarray, radius: int = 3):
    y_list, x_list = zip(*line)
    func = interpolate.interp1d(x_list, y_list)
    x_start, x_end = x_list[0], x_list[-1]
    x_list_new = np.arange(x_start, x_end)
    y_new = func(x_list_new)
    y_new_int = np.floor((y_new + 0.5)).astype(int)
    avg_blackness = 0
    for x in range(1, radius):
        y_step = y_new_int + x - 1
        indexes = (y_step, np.array(x_list_new))
        blackness = np.mean(image[indexes]) * 1.0 / x
        avg_blackness = avg_blackness + blackness
    for x in range(1, radius):
        y_step = y_new_int - x + 1
        indexes = (y_step, np.array(x_list_new))
        blackness = np.mean(image[indexes]) * 1.0 / x
        avg_blackness = avg_blackness + blackness
    return avg_blackness


def calculate_horizontal_runs(img: np.array, min_length: int):
    h = img.shape[0]
    w = img.shape[1]
    np_matrix = np.zeros([h, w], dtype=np.uint8)
    t = np.transpose(np.nonzero(np.diff(img) == -1))
    for trans in t:
        y, x = trans[0], trans[1] + 1
        xo = x
        # rl = 0
        while x < w and img[y, x] == 0:
            x += 1
        rl = x - xo
        if rl >= min_length:
            for x in range(xo, xo + rl):
                np_matrix[y, x] = 255
    return np_matrix


if __name__ == "__main__":
    from PIL import Image
    from matplotlib import pyplot as plt
    from linesegmentation.preprocessing.util import resize_image
    import os
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    page_path = os.path.join(project_dir, 'demo/images/test/line_test_image.jpg')

    def scale_line(line: Line, factor: float):
        return [[point[0] * factor, point[1] * factor] for point in line]

    factor_t = 4.0
    image_t = np.array(Image.open(page_path))
    line_t = [[130, 50], [133, 90], [134, 108], [136, 110], [131, 128], [138, 131], [139, 147], [142, 151], [142, 191],
            [140, 238], [149, 241], [149, 336], [150, 350], [142, 359], [148, 380], [152, 384], [142, 390], [152, 397],
              [140, 432], [177, 466], [195, 532], [188, 544], [200,  557], [205, 580], [193, 585], [205, 592],
              [191, 616], [185, 627]]
    image_cp = image_t.copy()
    scaled_image = resize_image(image_cp, factor_t)
    extent = (0, image_t.shape[1], image_t.shape[0], 0)
    line_a = np.asarray(line_t)
    y_p = line_a[:, 0]
    x_p = line_a[:, 1]

    yc, ax = plt.subplots(1, 2, True, True)
    ax[0].imshow(image_t, cmap="gray", extent=extent)
    ax[0].plot(x_p, y_p,)
    ax[0].plot(x_p, y_p, "bo")
    scaled_line = scale_line(line_a, factor_t)
    scaled_line = best_line_fit(scaled_image, scaled_line, line_thickness=3, max_iterations=30, scale=factor_t,
                                skip_startend_points=False)
    scale_line = scale_line(scaled_line, 1.0 / factor_t)
    line_a = np.asarray(scale_line)
    y_p = line_a[:, 0]
    x_p = line_a[:, 1]
    ax[1].imshow(image_t, cmap="gray", extent=extent)
    ax[1].plot(x_p, y_p)
    ax[1].plot(x_p, y_p, "bo")
    plt.show()


