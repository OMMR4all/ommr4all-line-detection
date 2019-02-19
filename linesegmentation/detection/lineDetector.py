# misc imports
import os
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import math
# project specific imports
from pagesegmentation.lib.predictor import PredictSettings
from scipy.interpolate import interpolate
from linesegmentation.pixelclassifier.predictor import PCPredictor
from PIL import Image
from linesegmentation.preprocessing.binarization.ocropus_binarizer import binarize
from linesegmentation.detection.lineDetectionUtil import vertical_runs
from linesegmentation.datatypes.datatypes import ImageData
from linesegmentation.util.image_util import smooth_array
from collections import defaultdict
from matplotlib import pyplot as plt


@dataclass
class LineDetectionSettings:
    numLine: int = 4
    minLineNum: int = 3
    minLength: int = 6
    lineExtension: bool = True
    debug: bool = False
    lineSpaceHeight: int = 20
    targetLineSpaceHeight: int = 10
    post_process: bool = False
    post_process_debug: bool = False
    smooth_lines: int = 0 # 0 = Off, 1 = basic Smoothing, 2 = Advanced Smoothing (slower)
    smooth_value_lowpass: float = 5
    smooth_value_adv: int = 25
    smooth_lines_advdebug: bool = False
    line_fit_distance: float = 0.5
    model: Optional[str] = None
    processes: int = 12


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


def create_data(path, line_space_height, load_image = False):
    space_height = line_space_height
    image = None
    if line_space_height == 0:
        image = np.array(Image.open(path)) / 255
        space_height = vertical_runs(binarize(image))[0]
    image_data = ImageData(path=path, height=space_height)
    if load_image:
        if line_space_height != 0:
            image = np.array(Image.open(path)) / 255
        image_data.image = image

    return image_data


class LineDetector():
    def __init__(self, settings: LineDetectionSettings):
        """Constructor of the LineDetector class

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

    def connect_connected_components_to_line(self, cc_list, staff_line_height, staff_space_height):
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
        return connect_cc(cc_list)

    def prune_small_lines(self, line_list, staff_space_height):
        medium_staff_height = [np.mean([y_c for y_c, x_c in staff]) for staff in line_list]
        line_list_copy = line_list.copy()
        while True:
            prev_staff_height = 0
            for staff_ind, staffh in enumerate(medium_staff_height):
                if (abs(prev_staff_height - staffh) < staff_space_height / 3.0) and prev_staff_height != 0:
                    y1a, x1a = line_list_copy[staff_ind - 1][0]
                    y1e, x1e = line_list_copy[staff_ind - 1][-1]
                    y2a, x2a = line_list_copy[staff_ind][0]
                    y2e, x2e = line_list_copy[staff_ind][-1]
                    if x2e >= x1e and x2a <= x1a:
                        del line_list_copy[staff_ind - 1]
                        del medium_staff_height[staff_ind - 1]
                        break
                    if x2e <= x1e and x2a >= x1a:
                        del line_list_copy[staff_ind]
                        del medium_staff_height[staff_ind]
                        break
                    if x2e >= x1e and x2a >= x1e:
                        line_list_copy[staff_ind - 1] = line_list_copy[staff_ind - 1] + line_list_copy[staff_ind]
                        del line_list_copy[staff_ind]
                        del medium_staff_height[staff_ind]
                        break
                    if x2e <= x1e and x1a >= x2e:
                        line_list_copy[staff_ind - 1] = line_list_copy[staff_ind] + line_list_copy[staff_ind - 1]
                        del line_list_copy[staff_ind]
                        del medium_staff_height[staff_ind]
                        break
                prev_staff_height = staffh
            else:
                break
        return line_list_copy

    def organize_lines_in_systems(self, line_list, staff_space_height, staff_line_height):
        # Calculate medium height of all staffs
        medium_staff_height = [np.mean([y_c for y_c, x_c in staff]) for staff in line_list]
        staff_indices = []
        for i, medium_y in enumerate(medium_staff_height):
            system = []
            if i in sum(staff_indices, []):
                continue
            height = medium_y
            for z, center_ys in enumerate(medium_staff_height):
                if np.abs(height - center_ys) < 1.3 * (staff_space_height + staff_line_height):
                    system.append(z)
                    height = center_ys
            staff_indices.append(system)
        staffindices = [staff for staff in staff_indices if len(staff) >= self.settings.minLineNum]
        staff_list = []
        for z in staffindices:
            system = []
            for x in z:
                system.append(line_list[x])
            staff_list.append(system)
        return staff_list

    def prune_lines_in_system_with_lowest_intensity(self, staff_list, img):
        ## Remove the lines with the lowest blackness value in each system, so that len(staffs) <= numLine
        prune = True
        while prune:
            prune = False
            for staff_ind, staff in enumerate(staff_list):
                if len(staff) > self.settings.numLine:
                    intensity_of_staff = {}
                    for line_ind, line in enumerate(staff):
                        intensity_of_staff[line_ind] = get_blackness_of_line(line , img)
                    if intensity_of_staff:
                        prune = True
                        min_blackness = min(intensity_of_staff.items(), key=lambda t: t[1])
                        if min_blackness[0] == 0 or min_blackness[0] == len(intensity_of_staff):
                            del staff_list[staff_ind][min_blackness[0]]
                            del intensity_of_staff[min_blackness[0]]
                            continue
                        if len(staff) >= self.settings.numLine * 2 + 1 and self.settings.numLine != 0:
                            if len(staff[:min_blackness[0]]) > 2:
                                staff_list.append(staff[:min_blackness[0]])
                            if len(staff[min_blackness[0]:]) > 2:
                                staff_list.append(staff[min_blackness[0]:])
                            del staff_list[staff_ind]
                            continue
                        del staff_list[staff_ind][min_blackness[0]]
                        del intensity_of_staff[min_blackness[0]]
        return staff_list

    def normalize_lines_in_system(self, staff_list, staff_space_height, img):
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
        return staff_list

    def postprocess_staff_systems(self, staffs_lines, line_height, image):
        post_processed_staff_systems = []
        h = image.shape[0]
        l2 = math.ceil(line_height / 2)
        l2 = max(l2, 2)
        for system in staffs_lines:
            procssed_system = []
            for staff in system:
                y, x = zip(*staff)
                f = interpolate.interp1d(x, y)
                x_start, x_end = min(x), max(x)
                dict_count = defaultdict(list)
                for i in range(x_start, x_end):

                    st_point = int(f(i))
                    if image[st_point][i] != 0:
                        for z in range(1, l2 + 1):
                            if image[st_point - z][i] == 0:
                                st_point = st_point - z
                                break
                            if image[st_point + z][i] == 0:
                                st_point = st_point + z
                                break
                    yt = st_point
                    yb = st_point

                    if image[yt][i] == 0:
                        dict_count[i].append(yt)
                        while yt < h - 1:
                            yt += 1
                            if image[yt][i] == 0:
                                dict_count[i].append(yt)
                            else:
                                break
                        while yb > 0:
                            yb -= 1
                            if image[yb][i] == 0:
                                dict_count[i].append(yb)
                            else:
                                break
                processed_staff = []
                for key in dict_count.keys():
                    if len(dict_count[key]) <= line_height:
                        processed_staff.append([np.mean(dict_count[key]), key])
                procssed_system.append(processed_staff)
            post_processed_staff_systems.append(procssed_system)

        if self.settings.post_process_debug:
            f, ax = plt.subplots(1, 2, True, True)
            ax[0].imshow(image, cmap='gray')
            ax[1].imshow(image, cmap='gray')
            cmap = plt.get_cmap('jet')
            colors = cmap(np.linspace(0, 1.0, len(staffs_lines)))
            for system, color in zip(staffs_lines, colors):
                for staff in system:
                    y, x = zip(*staff)
                    ax[0].plot(x, y, color=color)
            for system, color in zip(post_processed_staff_systems, colors):
                for staff in system:
                    y, x = zip(*staff)
                    ax[1].plot(x, y, color=color)
            plt.show()
        return post_processed_staff_systems

    def smooth_lines(self, stafflines, smooth_value=5):
        new_stafflines = []
        for system in stafflines:
            new_system = []
            for line in system:
                y, x = zip(*line)
                y = smooth_array(list(y), smooth_value)
                line = list(zip(y, x))
                new_system.append(line)
            new_stafflines.append(new_system)

        return new_stafflines

    def smooth_lines_advanced(self, stafflines, smooth_value=25):
        new_stafflines = []
        for system in stafflines:
            new_system = []
            for line in system:
                y, x = zip(*line)
                x = list(x)
                y = list(y)
                x, y = interpolate_sequence(x, y)
                append_start = [y[0] for x in range(10)]
                append_end = [y[-1] for x in range(10)]
                m_y = append_start + y + append_end
                remove_hill(m_y, smooth_value)
                line = list(zip(m_y[10:-10], x))
                new_system.append(line)
            new_stafflines.append(new_system)

        if self.settings.smooth_lines_advdebug:
            f, ax = plt.subplots(1, 2, True, True)
            cmap = plt.get_cmap('jet')
            colors = cmap(np.linspace(0, 1.0, len(new_stafflines)))
            for system, color in zip(stafflines, colors):
                for staff in system:
                    y, x = zip(*staff)
                    ax[0].plot(x, y, color=color)
            for system, color in zip(new_stafflines, colors):
                for staff in system:
                    y, x = zip(*staff)
                    ax[1].plot(x, y, color=color)
            plt.show()
        return new_stafflines


def remove_hill(y, smooth=25):
    # this will modify y
    correction = True
    it = 1
    while correction:
        correction = False
        it += 1

        for hlen in range(1, smooth + 1):
            for start in range(len(y) - (hlen + 1)):
                start_y = y[start]
                end_y = y[start + hlen + 1]

                # correct down
                for xc in range(start + 1, start + hlen + 1):
                    yc = y[xc]
                    if not (yc > start_y and yc > end_y):
                        break
                else:
                    for xc in range(start + 1, start + hlen + 1):
                        y[xc] = max(start_y, end_y)
                        correction = True
                if correction: break
                # correct up
                for xc in range(start + 1, start + hlen + 1):
                    yc = y[xc]
                    if not (yc < start_y and yc < end_y):
                        break
                else:
                    for xc in range(start + 1, start + hlen + 1):
                        y[xc] = min(start_y, end_y)
                        correction = True

                if correction: break
            if correction: break


def interpolate_sequence(x_list, y_list):
    func = interpolate.interp1d(x_list, y_list)
    x_start, x_end = x_list[0], x_list[-1]
    x_list_new = []
    y_list_new = []
    for i in range(x_start, x_end + 1):
        x_list_new.append(i)
        y_list_new.append(int(math.floor(func(i) + 0.5)))
    return x_list_new, y_list_new


def line_fitting(stafflist, dist=0.5):
    new_stafflist = []
    for system in stafflist:
        new_system = []
        for line in system:
            line = np.flip(np.asarray(line), axis=-1)
            simplified = ramerdouglas(line.tolist(), dist=dist)
            simplified = np.flip(np.asarray(simplified), axis=-1)
            new_system.append(simplified.tolist())
        new_stafflist.append(new_system)
    return new_stafflist


def _vec2d_dist(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2


def _vec2d_sub(p1, p2):
    return (p1[0]-p2[0], p1[1]-p2[1])


def _vec2d_mult(p1, p2):
    return p1[0]*p2[0] + p1[1]*p2[1]


def ramerdouglas(line, dist):
    """Does Ramer-Douglas-Peucker simplification of a curve with `dist`
    threshold.
    https://stackoverflow.com/questions/2573997/reduce-number-of-points-in-line
    `line` is a list-of-tuples, where each tuple is a 2D coordinate

    Usage is like so:

    myline = [(0.0, 0.0), (1.0, 2.0), (2.0, 1.0)]
    simplified = ramerdouglas(myline, dist = 1.0)
    """

    if len(line) < 3:
        return line

    (begin, end) = (line[0], line[-1]) if line[0] != line[-1] else (line[0], line[-2])

    distSq = []
    for curr in line[1:-1]:
        tmp = (
            _vec2d_dist(begin, curr) - _vec2d_mult(_vec2d_sub(end, begin), _vec2d_sub(curr, begin)) ** 2 / _vec2d_dist(begin, end))
        distSq.append(tmp)

    maxdist = max(distSq)
    if maxdist < dist ** 2:
        return [begin, end]

    pos = distSq.index(maxdist)
    return (ramerdouglas(line[:pos + 2], dist) +
            ramerdouglas(line[pos + 1:], dist)[1:])



if __name__ == "__main__":
    y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 9, 9, 9, 9, 9, 9]
    x = [i for i in range(len(y))]
    print(y)
    print(x)
