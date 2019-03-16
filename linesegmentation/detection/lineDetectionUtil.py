import peakutils
from scipy.signal import medfilt2d
import numpy as np
from itertools import tee


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
    white_r = np.argmax(white_runs[black_r * 2:]) + 1 + black_r * 2
    img = np.transpose(img)
    return white_r, black_r


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
