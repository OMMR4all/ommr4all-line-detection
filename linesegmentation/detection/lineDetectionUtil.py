import peakutils
from scipy.signal import medfilt2d
import numpy as np
from itertools import tee
from scipy.interpolate import interpolate
from numpy.linalg import norm


def get_text_borders(image, preprocess=False, min_dist=30, thres=0.3):
    med = image.copy()
    if preprocess:
        med = medfilt2d(image, 9)
    histogram = np.sum(med == 255, axis=1)
    text_borders = peakutils.indexes(histogram, thres=thres, min_dist=min_dist)
    return text_borders


def vertical_runs(img: np.array):
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


def angle_difference_of_points(x1, y1, x2, y2):
    v1 = np.array([x1, y1])
    v2 = np.array([x2, y2])

    angle_difference = np.arccos((v1 @ v2) / (norm(v1) * norm(v2)))

    return angle_difference


def simplify_anchor_points(line, max_distance=25, min_distance=10, min_degree_to_keep_points=0.2):
    new_line = []

    def distance(p1, p2):
        return p2[1] - p1[1]
    prev_point = None
    for point_ind, point in enumerate(line):
        if prev_point is not None:
            point_distance = distance(prev_point, point)
        else:
            point_distance = min_distance + 1
        if point_distance > max_distance:
            new_line.append([point[0], prev_point[1] + (point[1] - prev_point[1])/2])
        if point_distance < min_distance:
            if angle_difference_of_points(prev_point[1], prev_point[0], point[1],
                                          point[0]) > min_degree_to_keep_points or point_ind == len(line) - 1:
                new_line.append(point)
        else:
            new_line.append(point)
        prev_point = point
    return new_line


def best_line_fit(img: np.array, line, line_thickness=3, max_iterations=30, scale=1.0, skip_startend_points=False):
    current_blackness = get_blackness_of_line(line, img)
    best_line = line.copy()
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
            y, x = point[0], point[1]
            for i in range(1, line_thickness * np.ceil(scale).astype(int)):
                test_line = best_line.copy()
                test_line[point_ind] = [y + i, x]
                blackness = get_blackness_of_line(test_line, img)

                if blackness < current_blackness:
                    change = True
                    current_blackness = blackness
                    best_line[point_ind] = [y + i, x]

                test_line[point_ind] = [y - i, x]
                blackness = get_blackness_of_line(test_line, img)

                if blackness < current_blackness:
                    change = True
                    current_blackness = blackness
                    best_line[point_ind] = [y - i, x]

        iterations += 1
    return best_line


def get_blackness_of_line(line, image):
    y_list, x_list = zip(*line)
    func = interpolate.interp1d(x_list, y_list)
    x_start, x_end = int(x_list[0]), int(x_list[-1])
    x_list_new = np.asarray(list(range(x_start, x_end)))
    y_new = func(x_list_new)
    y_new_int = np.floor(y_new + 0.5).astype(int)
    indexes = (np.array(y_new_int), np.array(x_list_new))
    blackness = np.mean(image[indexes])
    return blackness


# Used for testing
def get_blackness_of_line_distribution(line, image, radius=3):
    y_list, x_list = zip(*line)
    func = interpolate.interp1d(x_list, y_list)
    x_start, x_end = x_list[0], x_list[-1]
    x_list_new = np.asarray(list(range(x_start, x_end)))
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


def scale_line(line, factor):
    return [[point[0] * factor, point[1] * factor] for point in line]


if __name__ == "__main__":
    from PIL import Image
    from matplotlib import pyplot as plt
    from linesegmentation.preprocessing.preprocessingUtil import resize_image
    import os
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    page_path = os.path.join(project_dir, 'demo/images/test/line_test_image.jpg')

    factor = 2.0
    image = np.array(Image.open(page_path))
    line = [[130, 50], [133, 90], [134, 108], [136, 110], [131, 128], [138, 131], [139, 147], [142, 151], [142, 191],
            [140, 238], [149, 241], [149, 336]]
    image_cp = image.copy()
    scaled_image = resize_image(image_cp, factor)
    y, x = zip(*line)
    yc, ax = plt.subplots(1, 2, True, True)
    ax[0].imshow(image, cmap="gray")
    ax[0].plot(x, y,)
    ax[0].plot(x, y, "bo")
    scaled_line = scale_line(line, factor)
    scaled_line = best_line_fit(scaled_image, scaled_line, line_thickness=3, max_iterations=30, scale=factor,
                                skip_startend_points=False)
    scale_line = scale_line(scaled_line, 1.0 / factor)

    y, x = zip(*scale_line)
    ax[1].imshow(image, cmap="gray")
    ax[1].plot(x, y,)
    ax[1].plot(x, y, "bo")
    plt.show()
