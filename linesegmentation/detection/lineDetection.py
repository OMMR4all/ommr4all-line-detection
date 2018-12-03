# misc imports
import cv2
import multiprocessing
import operator
import os
import tqdm
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from itertools import tee
from typing import List, Generator

import numpy as np
# image specific imports
from PIL import Image
from matplotlib import pyplot as plt
# project specific imports
from pagesegmentation.lib.predictor import PredictSettings
from scipy.interpolate import interpolate
from scipy.ndimage.morphology import binary_erosion, binary_dilation

from linesegmentation.pixelclassifier.predictor import PCPredictor
from linesegmentation.preprocessing.binarization.ocropus_binarizer import binarize
from linesegmentation.preprocessing.enhancing.enhancer import enhance


@dataclass
class ImageData:
    path: str = None
    height: int = None
    image: np.array = None
    horizontal_runs_img: np.array = None
    staff_line_height: int = None
    staff_space_height: int = None


@dataclass
class LineDetectionSettings:
    numLine: int = 4
    minLineNum: int = 3
    minLength: int = 6
    lineExtension: bool = True
    debug: bool = False
    lineSpaceHeight: int = 20
    targetLineSpaceHeight: int = 10
    model: Optional[str] = None
    processes: int = 12


def create_data(path, line_space_height):
    space_height = line_space_height
    if line_space_height == 0:
        space_height = vertical_runs(binarize(np.array(Image.open(path)) / 255))[0]
    image_data = ImageData(path=path, height=space_height)
    return image_data


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
    # --> skip the first elements of the array
    white_r = np.argmax(white_runs[black_r:]) + 1 + black_r
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


class LineDetection:
    """Line detection class

    Attributes
    ----------
    settings : LineDetectionSettings
        Setting for the line detection algorithm
    predictor : PCPredictor, optional
        Necessary if the NN should be used for the binarisation

    """

    def __init__(self, settings: LineDetectionSettings):
        """Constructor of the LineDetection class

        Parameters
        ----------
        settings: LineDetectionSettings
            Settings for the line detection algorithm
        """
        self.settings = settings
        self.predictor = None
        if settings.model:
            pcsettings = PredictSettings(
                mode='meta',
                network=os.path.abspath(settings.model),
                output=None,
                high_res_output=False
            )
            self.predictor = PCPredictor(pcsettings, settings.targetLineSpaceHeight)

    def detect(self, image_paths: List[str]) -> Generator[List[List[List[int]]], None, None]:
        """
        Function  to detect die stafflines in an image

        Parameters
        ----------
        image_paths: List[str]
            Paths to the images, which should be processed

        Yields
        ------
        List     [List    [List      [int]]]
        System   Staff    Polyline    y,x
        
            Example
            --------
            ####### Structure ######
            pointList[
                       system1[
                              staff1[
                                   [y1, x1]
                                   [y2, x2]
                                   ]
                              staff2[
                                     ...
                                   ]
                       system2[
                               ...
                             ]
                     ]    
        """
        if not self.settings.model:
            return self.detect_basic(image_paths)
        else:
            return self.detect_advanced(image_paths)

    def detect_basic(self, image_paths: List[str]) -> Generator[List[List[List[int]]], None, None]:
        for img_path in image_paths:
            image_data = ImageData(path=img_path)
            image_data.image = np.array(Image.open(img_path)) / 255
            gray = image_data.image
            if np.sum(np.histogram(gray)[0][1:-2]) != 0:
                gray = enhance(image_data.image)
            binary = binarize(gray)
            binarized = 1 - binary
            morph = binary_erosion(binarized, structure=np.full((5, 1), 1))
            morph = binary_dilation(morph, structure=np.full((5, 1), 1))
            staffs = (binarized ^ morph)
            image_data.staff_space_height, image_data.staff_line_height = vertical_runs(binary)
            image_data.horizontal_runs_img = calculate_horizontal_runs((1 - staffs), self.settings.minLength)
            yield self.detect_staff_lines(image_data)

    def detect_advanced(self, image_paths: List[str]) -> Generator[List[List[List[int]]], None, None]:

        create_data_partital = partial(create_data, lineSpaceHeight=self.settings.lineSpaceHeight)
        with multiprocessing.Pool(processes=self.settings.processes) as p:
            data = [v for v in tqdm.tqdm(p.imap(create_data_partital, image_paths), total=len(image_paths))]

        for i, pred in enumerate(self.predictor.predict(data)):
            pred[pred > 0] = 255
            data[i].staff_space_height, data[i].staff_line_height = vertical_runs(1 - pred)
            data[i].horizontal_runs_img = calculate_horizontal_runs((1 - (pred / 255)), self.settings.minLength)
            yield self.detect_staff_lines(data[i])

    def detect_staff_lines(self, image_data: ImageData):
        img = image_data.horizontal_runs_img
        staff_line_height = image_data.staff_line_height
        staff_space_height = image_data.staff_space_height
        connectivity = 8
        output = cv2.connectedComponentsWithStats(img, connectivity)

        # Normalize the CCs (line segments), so that the height of each cc is normalized to one pixel
        def normalize(point_list):
            normalized_cc_list = []
            for cc in point_list:
                cc_dict = defaultdict(list)
                for y, x in cc:
                    cc_dict[x].append(y)
                normalized_cc = []
                for key, value in cc_dict.items():
                    normalized_cc.append([int(np.floor(np.mean(value) + 0.5)), key])
                normalized_cc_list.append(normalized_cc)
            return normalized_cc_list

        # Calculate the position, sort and normalize all CCs
        ccdict = defaultdict(list)
        indexdim0, indexdim1 = np.array(output[1]).nonzero()
        points = list(zip(indexdim0, indexdim1))
        for p in points:
            y_coord, x_coord = p[0], p[1]
            k = output[1][y_coord][x_coord]
            ccdict[k].append([y_coord, x_coord])
        cc_list = list(ccdict.values())
        [x.sort(key=operator.itemgetter(1)) for x in cc_list]
        cc_list = normalize(cc_list)

        def connect_cc(cc_list, inplace=True):
            def prune_cc(cc_list, length):
                pruned_cc_list = []
                for cc in cc_list:
                    if abs(cc[0][1] - cc[-1][1]) > length:
                        pruned_cc_list.append(cc)
                return pruned_cc_list

            def connect(max_dists: List[int], vert_dist: int, cc_list):
                for max_dist in max_dists:
                    i = 0
                    while i < len(cc_list):
                        l1 = cc_list[i]
                        y1b, x1b = l1[0]
                        y1e, x1e = l1[-1]

                        found = False
                        for i2 in range(i + 1, len(cc_list)):
                            l2 = cc_list[i2]
                            y2b, x2b = l2[0]
                            y2e, x2e = l2[-1]
                            if x1e < x2b and x2b - x1e < max_dist:
                                if np.abs(y1e - y2b) < vert_dist:
                                    cc_list[i] = l1 + l2
                                    del cc_list[i2]
                                    found = True
                                    break
                            elif x2e < x1b and x1b - x2e < max_dist:
                                if np.abs(y1b - y2e) < vert_dist:
                                    cc_list[i] = l2 + l1
                                    del cc_list[i2]
                                    found = True
                                    break
                        if not found:
                            i += 1
                    if vert_dist == 2 and max_dist == 30:
                        cc_list = prune_cc(cc_list, 10)
                return cc_list

            cc_list_copy = cc_list

            if not inplace:
                cc_list_copy = cc_list.copy()

            for x in [[10, 30, 50, 100], [200, 300, 500]]:
                for vert_dist in [2, staff_line_height, staff_space_height / 5 + staff_line_height,
                                  staff_space_height / 3 + staff_line_height]:
                    cc_list_copy = connect(x, vert_dist, cc_list_copy)
            return cc_list_copy

        line_list = connect_cc(cc_list)
        # Remove lines which are shorter than 50px

        line_list = [l for l in line_list if l[-1][1] - l[0][1] > 50]

        # Calculate medium height of all staffs
        medium_staff_height = [np.mean([y_c for y_c, x_c in staff]) for staff in line_list]

        # Debug
        staff2 = line_list.copy()

        def prune_small_lines(line_list, medium_staff_height, inplace=True):
            line_list_copy = line_list
            medium_staff_height_copy = medium_staff_height
            if not inplace:
                medium_staff_height_copy = medium_staff_height.copy()
                line_list_copy = line_list_copy.copy()
            while True:
                prev_staff_height = 0
                for staff_ind, staffh in enumerate(medium_staff_height_copy):
                    if (abs(prev_staff_height - staffh) < staff_space_height / 3.0) and prev_staff_height != 0:
                        y1a, x1a = line_list_copy[staff_ind - 1][0]
                        y1e, x1e = line_list_copy[staff_ind - 1][-1]
                        y2a, x2a = line_list_copy[staff_ind][0]
                        y2e, x2e = line_list_copy[staff_ind][-1]
                        if x2e >= x1e and x2a <= x1a:
                            del line_list_copy[staff_ind - 1]
                            del medium_staff_height_copy[staff_ind - 1]
                            break
                        if x2e <= x1e and x2a >= x1a:
                            del line_list_copy[staff_ind]
                            del medium_staff_height_copy[staff_ind]
                            break
                        if x2e >= x1e and x2a >= x1e:
                            line_list_copy[staff_ind - 1] = line_list_copy[staff_ind - 1] + line_list_copy[staff_ind]
                            del line_list_copy[staff_ind]
                            del medium_staff_height_copy[staff_ind]
                            break
                        if x2e <= x1e and x1a >= x2e:
                            line_list_copy[staff_ind - 1] = line_list_copy[staff_ind] + line_list_copy[staff_ind - 1]
                            del line_list_copy[staff_ind]
                            del medium_staff_height_copy[staff_ind]
                            break
                    prev_staff_height = staffh
                else:
                    break
            return line_list_copy, medium_staff_height_copy

        line_list, medium_staff_height = prune_small_lines(line_list, medium_staff_height, inplace=True)

        if self.settings.numLine > 1:
            staffindices = []
            for i, medium_y in enumerate(medium_staff_height):
                system = []
                if i in sum(staffindices, []):
                    continue
                height = medium_y
                for z, center_ys in enumerate(medium_staff_height):
                    if np.abs(height - center_ys) < 1.3 * (staff_space_height + staff_line_height):
                        system.append(z)
                        height = center_ys
                staffindices.append(system)
            staffindices = [staff for staff in staffindices if len(staff) >= self.settings.minLineNum]

            def get_blackness_of_line(line, image):
                y_list, x_list = zip(*line)
                func = interpolate.interp1d(x_list, y_list)
                x_start, x_end = x_list[0], x_list[-1]
                spaced_numbers = np.linspace(x_start, x_end, num=int(abs(x_list[0] - x_list[-1]) * 1 / 5), endpoint=True)
                blackness = 0
                for number in spaced_numbers:
                    if image[int(func(number))][int(number)] == 255:
                        blackness += 1
                return blackness

            ## Remove the lines with the lowest blackness value in each system, so that len(staffs) <= numLine
            prune = True
            while prune:
                prune = False
                for staff_ind, staff in enumerate(staffindices):
                    if len(staff) > self.settings.numLine:
                        intensity_of_staff = {}
                        for line_ind, line in enumerate(staff):
                            intensity_of_staff[line_ind] = get_blackness_of_line(line_list[line], img)
                        if intensity_of_staff:
                            prune = True
                            min_blackness = min(intensity_of_staff.items(), key=lambda t: t[1])
                            if min_blackness[0] == 0 or min_blackness[0] == len(intensity_of_staff):
                                del staffindices[staff_ind][min_blackness[0]]
                                del intensity_of_staff[min_blackness[0]]
                                continue
                            if len(staff) >= self.settings.numLine * 2 + 1 and self.settings.numLine != 0:
                                if len(staff[:min_blackness[0]]) > 2:
                                    staffindices.append(staff[:min_blackness[0]])
                                if len(staff[min_blackness[0]:]) > 2:
                                    staffindices.append(staff[min_blackness[0]:])
                                del staffindices[staff_ind]
                                continue
                            del staffindices[staff_ind][min_blackness[0]]
                            del intensity_of_staff[min_blackness[0]]

            staff_list = []
            for z in staffindices:
                system = []
                for x in z:
                    system.append(line_list[x])
                staff_list.append(system)

            if self.settings.lineExtension:

                for z_ind, z in enumerate(staff_list):
                    sxs = [line[0][1] for line in z]
                    exs = [line[-1][1] for line in z]
                    min_index_sxs, sxb = sxs.index(min(sxs)), min(sxs)
                    max_index_exs, exb = exs.index(max(exs)), max(exs)
                    ymi, xmi = zip(*z[min_index_sxs])
                    minf = interpolate.interp1d(xmi, ymi, fill_value='extrapolate')
                    yma, xma = zip(*z[max_index_exs])
                    maxf = interpolate.interp1d(xma, yma, fill_value='extrapolate')

                    for line_ind, line in enumerate(z):
                        y, x = zip(*line)
                        if line[0][1] > xmi[0] and abs(line[0][1] - xmi[0]) > 5:
                            x_start, x_end = xmi[0], min(line[0][1], z[min_index_sxs][-1][1])
                            spaced_numbers = np.linspace(x_start, x_end - 1, num=abs(x_end - x_start) * 1 / 5, endpoint=True)
                            staffextension = []
                            if line[0][1] > xmi[-1]:
                                dif = minf(xma[-1]) - line[0][0]
                            else:
                                dif = minf(line[0][1]) - line[0][0]
                            for i in spaced_numbers:
                                staffextension.append([int(minf(i) - dif), int(i)])
                            if staffextension:
                                staff_list[z_ind][line_ind] = staffextension + staff_list[z_ind][line_ind]
                        if line[-1][1] < exs[max_index_exs] and abs(line[-1][1] - exs[max_index_exs]) > 5:
                            x_start, x_end = max(line[-1][1], z[max_index_exs][0][1]), exs[max_index_exs]
                            spaced_numbers = np.linspace(x_start, x_end, num=abs(x_end - x_start) * 1 / 5, endpoint=True)
                            staffextension = []
                            if line[-1][1] < xma[0]:
                                dif = maxf(xma[0]) - line[-1][0]
                            else:
                                dif = maxf(line[-1][1]) - line[-1][0]
                            for i in spaced_numbers:
                                staffextension.append([int(maxf(i) - dif), int(i)])
                            if staffextension:
                                staff_list[z_ind][line_ind] = staff_list[z_ind][line_ind] + staffextension
                        if line[0][1] < sxb and abs(line[0][1] - sxs[min_index_sxs]) > 5:
                            while len(line) > 0 and line[0][1] <= sxb:
                                del line[0]
                        if x[-1] > exb and abs(x[-1] - sxs[min_index_sxs]) > 5:
                            while line[-1][1] >= exb:
                                del line[-1]

                for staff_ind, staffs in enumerate(staff_list):
                    medium_staff_height_of_line = [np.mean([y for y, x in line]) for line in staffs]
                    while True:
                        prev_line_height = 0
                        for line_ind, lineh in enumerate(medium_staff_height_of_line):
                            if (abs(prev_line_height - lineh) < staff_space_height / 2.0) and prev_line_height != 0:
                                blackness1 = get_blackness_of_line(staff_list[staff_ind][line_ind], img)
                                blackness2 = get_blackness_of_line(staff_list[staff_ind][line_ind - 1], img)
                                if blackness1 > blackness2:
                                    del staff_list[staff_ind][line_ind - 1]
                                    del medium_staff_height_of_line[line_ind - 1]
                                    break
                                else:
                                    del staff_list[staff_ind][line_ind]
                                    del medium_staff_height_of_line[line_ind]
                                    break
                            prev_line_height = lineh
                        else:
                            break
        else:
            staff_list = line_list
        # Debug
        if self.settings.debug:
            im = plt.imread(image_data.path)
            f, ax = plt.subplots(1, 3, True, True)
            ax[0].imshow(im, cmap='gray')
            cmap = plt.get_cmap('jet')
            colors = cmap(np.linspace(0, 1.0, len(staff_list)))
            for system, color in zip(staff_list, colors):
                for staff in system:
                    y, x = zip(*staff)
                    ax[0].plot(x, y, color=color)
            ax[1].imshow(img, cmap='gray')
            ax[2].imshow(im, cmap='gray')
            for staff in staff2:
                y, x = zip(*staff)
                ax[2].plot(x, y, 'r')
            plt.show()
        return staff_list


if __name__ == "__main__":
    setting = LineDetectionSettings(debug=True)
    line_detector = LineDetection(setting)
    for pred in line_detector.detect(['/home/alexanderh/Schreibtisch/masterarbeit/OMR/Graduel_de_leglise_de_Nevers/interesting/part1/bin/Graduel_de_leglise_de_Nevers-035.nrm.png']):
        pass
