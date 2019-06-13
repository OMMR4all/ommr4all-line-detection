# misc imports
import os
from typing import List, Optional, NamedTuple, Tuple
import numpy as np
import math
# project specific imports
from pagesegmentation.lib.predictor import PredictSettings
from scipy.interpolate import interpolate
from linesegmentation.pixelclassifier.predictor import PCPredictor
from linesegmentation.detection.util import vertical_runs, best_line_fit, get_blackness_of_line,\
    simplify_anchor_points
from linesegmentation.datatypes.datatypes import ImageData
from linesegmentation.util.image_util import smooth_array
from collections import defaultdict
from matplotlib import pyplot as plt
from linesegmentation.preprocessing.binarization.basic_binarizer import gauss_threshold
from linesegmentation.preprocessing.util import resize_image
from linesegmentation.preprocessing.polysimplify import VWSimplifier
from linesegmentation.detection.settings import LineDetectionSettings, LineSimplificationAlgorithm
from linesegmentation.preprocessing.util import extract_connected_components, \
    normalize_connected_components
from linesegmentation.detection.datatypes import Line, Point, System


class LineDetector:
    def __init__(self, settings: LineDetectionSettings):
        """Constructor of the LineDetector class

        Parameters
        ----------
        settings: LineDetectionSettings
            Settings for the line detection algorithm

        Attributes
        ----------
        settings: LineDetectionSettings
            Settings for the line detection algorithm
        predictor : PCPredictor, optional
             Necessary if the NN should be used for the binarisation
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
            self.predictor = PCPredictor(pcsettings, settings.target_line_space_height)

    @staticmethod
    def extract_ccs(img: np.ndarray) -> List[List[Point]]:
        cc_list = extract_connected_components(img)
        cc_list = normalize_connected_components(cc_list)
        cc_list_new = []
        for cc in cc_list:
            cc_new = []
            for y, x in cc:
                cc_new.append(Point(x, y))
            cc_list_new.append(cc_new)
        return cc_list_new

    @staticmethod
    def connect_connected_components_to_line(cc_list: List[List[Point]], staff_line_height: int,
                                             staff_space_height: int) -> List[Line]:

        def connect_cc(cc_list: List[List[List[int]]]):
            def prune_cc(cc_list: List[List[Point]], length: int):
                pruned_cc_list = []
                for cc in cc_list:
                    if abs(cc[0].x - cc[-1].x) > length:
                        pruned_cc_list.append(cc)
                return pruned_cc_list

            def connect(max_dists: List[int], vert_dist: int, cc_list: List[List[Point]]):
                for max_dist in max_dists:
                    i = 0
                    while i < len(cc_list):
                        l1 = cc_list[i]
                        p1_b = l1[0]
                        p1_e = l1[-1]

                        found = False
                        for i2 in range(i + 1, len(cc_list)):
                            l2 = cc_list[i2]
                            p2_b = l2[0]
                            p2_e = l2[-1]
                            if p1_e.x < p2_b.x and p2_b.x - p1_e.x < max_dist:
                                if np.abs(p1_e.y - p2_b.y) < vert_dist:
                                    cc_list[i] = l1 + l2
                                    del cc_list[i2]
                                    found = True
                                    break
                            elif p2_e.x < p1_b.x and p1_b.x - p2_e.x < max_dist:
                                if np.abs(p1_b.y - p2_e.y) < vert_dist:
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

            for x in [[10, 30, 50, 100], [200, 300, 500]]:
                for vert_dist in [2, staff_line_height, staff_space_height / 5 + staff_line_height,
                                  staff_space_height / 3 + staff_line_height]:
                    cc_list_copy = connect(x, vert_dist, cc_list_copy)
            return cc_list_copy

        llc = connect_cc(cc_list)
        lines = [Line(points) for points in llc]
        return lines

    @staticmethod
    def prune_small_lines(line_list: List[Line], staff_space_height: int) -> List[Line]:
        mean_line_height_list = [line.get_average_line_height() for line in line_list]
        line_list_copy = line_list.copy()
        while True:
            prev_line_height = 0
            for line_ind, line_height in enumerate(mean_line_height_list):
                if (abs(prev_line_height - line_height) < staff_space_height / 3.0) and prev_line_height != 0:
                    p1a = line_list_copy[line_ind - 1].get_start_point()
                    p1e = line_list_copy[line_ind - 1].get_end_point()
                    p2a = line_list_copy[line_ind].get_start_point()
                    p2e = line_list_copy[line_ind].get_end_point()
                    if p2e.x >= p1e.x and p2a.x <= p1a.x:
                        del line_list_copy[line_ind - 1]
                        del mean_line_height_list[line_ind - 1]
                        break
                    if p2e.x <= p1e.x and p2a.x >= p1a.x:
                        del line_list_copy[line_ind]
                        del mean_line_height_list[line_ind]
                        break
                    if p2e.x >= p1e.x and p2a.x >= p1a.x:
                        line_list_copy[line_ind - 1] = Line(line_list_copy[line_ind - 1] + line_list_copy[line_ind])
                        del line_list_copy[line_ind]
                        del mean_line_height_list[line_ind]
                        break
                    if p2e.x <= p1e.x and p1a.x >= p2e.x:
                        line_list_copy[line_ind - 1] = Line(line_list_copy[line_ind] + line_list_copy[line_ind - 1])
                        del line_list_copy[line_ind]
                        del mean_line_height_list[line_ind]
                        break
                prev_line_height = line_height
            else:
                break
        return line_list_copy

    def organize_lines_in_systems(self, line_list: List[Line], staff_space_height: int,
                                  staff_line_height: int) -> List[System]:
        # Calculate medium height of all staffs
        mean_line_height_list = [line.get_average_line_height() for line in line_list]
        staff_indices = []
        for i, medium_y in enumerate(mean_line_height_list):
            system = []
            if i in sum(staff_indices, []):
                continue
            height = medium_y
            for z, center_ys in enumerate(mean_line_height_list):
                if np.abs(height - center_ys) < 1.3 * (staff_space_height + staff_line_height):
                    system.append(z)
                    height = center_ys
            staff_indices.append(system)
        staffindices = [staff for staff in staff_indices if len(staff) >= self.settings.min_lines_per_system]
        staff_list = []
        for z in staffindices:
            system_list = []
            for xt in z:
                system_list.append(line_list[xt])
            staff_list.append(System(system_list))
        return staff_list

    def prune_lines_in_system_with_lowest_intensity(self, system_list: List[System], img: np.ndarray) -> List[System]:
        # Remove the lines with the lowest blackness value in each system, so that len(staffs) <= numLine
        prune = True
        while prune:
            prune = False
            for system_ind, system in enumerate(system_list):
                if len(system) > self.settings.line_number:
                    intensity_of_line = {}
                    for line_ind, line in enumerate(system):
                        intensity_of_line[line_ind] = approximate_blackness_of_line(line, img)
                    if intensity_of_line:
                        prune = True
                        min_blackness: Tuple[int] = min(intensity_of_line.items(), key=lambda t: t[1])
                        if min_blackness[0] == 0 or min_blackness[0] == len(intensity_of_line):
                            del system_list[system_ind][min_blackness[0]]
                            del intensity_of_line[min_blackness[0]]
                            continue
                        if len(system) >= self.settings.line_number * 2 + 1 and self.settings.line_number != 0:
                            if len(system[:min_blackness[0]]) > 2:
                                system_list.append(system[:min_blackness[0]])
                            if len(system[min_blackness[0]:]) > 2:
                                system_list.append(system[min_blackness[0]:])
                            del system_list[system_ind]
                            continue
                        del system_list[system_ind][min_blackness[0]]
                        del intensity_of_line[min_blackness[0]]
        return system_list

    @staticmethod
    def normalize_lines_in_system(system_list: List[System], staff_space_height: int, img: np.ndarray) -> List[System]:
        for z_ind, z in enumerate(system_list):
            sxs = [line.get_start_point().x for line in z]
            exs = [line.get_end_point().x for line in z]

            min_index_sxs, sxb = sxs.index(min(sxs)), min(sxs)
            max_index_exs, exb = exs.index(max(exs)), max(exs)

            xmi, ymi = z[min_index_sxs].get_xy()
            xma, yma = z[max_index_exs].get_xy()
            if len(xmi) < 2 or len(xma) < 2:
                continue

            minf = interpolate.interp1d(xmi, ymi, fill_value=(ymi[0], ymi[-1]), bounds_error=False)
            maxf = interpolate.interp1d(xma, yma, fill_value=(yma[0], yma[-1]), bounds_error=False)

            for line_ind, line in enumerate(z):
                x, y = line.get_xy()
                point_start_x = line.get_start_point().x
                point_start_y = line.get_start_point().y
                point_end_x = line.get_end_point().x
                point_end_y = line.get_end_point().y
                if point_start_x > xmi[0] and abs(point_start_x - xmi[0]) > 5:
                    x_start, x_end = xmi[0], min(point_start_x, z[min_index_sxs].get_end_point().x)
                    x_values = list(range(x_start, x_end))

                    if point_start_x > xmi[-1]:
                        dif = minf(xma[-1]) - point_start_y
                    else:
                        dif = minf(point_start_x) - point_start_y
                    interpolated_ys = np.floor(minf(x_values) + 0.5 - dif)
                    interpolated_line_segment = [Point(x, y) for x, y in zip(x_values, interpolated_ys)]
                    if interpolated_line_segment:
                        system_list[z_ind][line_ind].l_append(interpolated_line_segment)

                if point_end_x < exs[max_index_exs] and abs(point_end_x - exs[max_index_exs]) > 5:

                    x_start, x_end = max(point_end_x, z[max_index_exs].get_start_point().x), exs[max_index_exs]
                    x_values = list(range(x_start, x_end))

                    if point_end_x < xma[0]:
                        dif = maxf(xma[0]) - point_end_y
                    else:
                        dif = maxf(point_end_x) - point_end_y

                    interpolated_ys = np.floor(maxf(x_values) + 0.5 - dif)
                    interpolated_line_segment = [Point(x, y) for x, y in zip(x_values, interpolated_ys)]

                    if interpolated_line_segment:
                        system_list[z_ind][line_ind].r_append(interpolated_line_segment)
                if point_start_x < sxb and abs(point_end_y - sxs[min_index_sxs]) > 5:
                    while len(line) > 0 and point_end_y <= sxb:
                        del line[0]
                if x[-1] > exb and abs(x[-1] - sxs[min_index_sxs]) > 5:
                    while point_end_x >= exb:
                        del line[-1]

        for system_ind, system in enumerate(system_list):
            mean_line_height = [line.get_average_line_height() for line in system]
            while True:
                prev_line_height = 0
                for line_ind, lineh in enumerate(mean_line_height):
                    if (abs(prev_line_height - lineh) < staff_space_height / 2.0) and prev_line_height != 0:
                        blackness1 = get_blackness_of_line(system_list[system_ind][line_ind], img)
                        blackness2 = get_blackness_of_line(system_list[system_ind][line_ind - 1], img)
                        if blackness1 > blackness2:
                            del system_list[system_ind][line_ind - 1]
                            del mean_line_height[line_ind - 1]
                            break
                        else:
                            del system_list[system_ind][line_ind]
                            del mean_line_height[line_ind]
                            break
                    prev_line_height = lineh
                else:
                    break
        return system_list

    def post_process_staff_systems(self, staffs_lines: List[System], line_height: int, image: np.ndarray)\
            -> List[System]:
        post_processed_staff_systems = []
        h = image.shape[0]
        l2 = math.ceil(line_height / 2)
        l2 = max(l2, 2)
        for system in staffs_lines:
            processed_system = []
            for staff in system:
                x, y = staff.get_xy()
                x_new, y_new = interpolate_sequence(x, y)
                dict_count = defaultdict(list)
                for i_ind, i in enumerate(y_new):
                    st_point = min(i, image.shape[0] - 1)

                    max_l2 = min(abs(st_point - image.shape[0]), l2 + 1)
                    if image[st_point][x_new[i_ind]] != 0:
                        for z in range(1, max_l2):
                            if image[st_point - z][x_new[i_ind]] == 0:
                                st_point = st_point - z
                                break
                            if image[st_point + z][x_new[i_ind]] == 0:
                                st_point = st_point + z
                                break
                    yt = st_point
                    yb = st_point

                    if image[yt][x_new[i_ind]] == 0:
                        dict_count[x_new[i_ind]].append(yt)
                        while yt < h - 1:
                            yt += 1
                            if image[yt][x_new[i_ind]] == 0:
                                dict_count[x_new[i_ind]].append(yt)
                            else:
                                break
                        while yb > 0:
                            yb -= 1
                            if image[yb][x_new[i_ind]] == 0:
                                dict_count[x_new[i_ind]].append(yb)
                            else:
                                break
                processed_staff = []
                for key in dict_count.keys():
                    if len(dict_count[key]) <= line_height:
                        processed_staff.append(Point(y=np.mean(dict_count[key]), x=key))
                processed_system.append(Line(processed_staff))
            post_processed_staff_systems.append(System(processed_system))

        for system_ind, system in enumerate(post_processed_staff_systems):
            post_processed_staff_systems[system_ind] = [lin for lin in system if lin]
        post_processed_staff_systems = [sys for sys in post_processed_staff_systems if sys]

        if self.settings.post_process_debug:
            f, ax = plt.subplots(1, 2, True, True)
            ax[0].imshow(image, cmap='gray')
            ax[1].imshow(image, cmap='gray')
            cmap = plt.get_cmap('jet')
            colors = cmap(np.linspace(0, 1.0, len(staffs_lines)))
            for system, color in zip(staffs_lines, colors):
                for staff in system:
                    x, y = staff.get_xy()
                    ax[0].plot(x, y, color=color)
            for system, color in zip(post_processed_staff_systems, colors):
                for staff in system:
                    x, y = staff.get_xy()
                    ax[1].plot(x, y, color=color)
            plt.show()
        return post_processed_staff_systems

    def smooth_lines(self, staff_lines: List[System]) -> List[System]:
        new_staff_lines = []
        for system in staff_lines:
            new_system = []
            for line in system:
                x, y = line.get_xy()
                y = smooth_array(list(y), self.settings.smooth_value_low_pass)

                line = Line([Point(x, y) for x, y in zip(x, y)])
                new_system.append(line)
            new_staff_lines.append(System(new_system))

        return new_staff_lines

    def smooth_lines_advanced(self, staff_lines: List[System]) -> List[System]:
        new_staff_lines = []
        for system in staff_lines:
            new_system = []
            for line in system:
                x, y = line.get_xy()
                if len(x) < 2:
                    continue

                x, y = interpolate_sequence(x, y)
                append_start = [y[0] for x in range(10)]
                append_end = [y[-1] for x in range(10)]
                m_y = append_start + y + append_end
                remove_hill(m_y, self.settings.smooth_value_adv)
                line = Line([Point(x, y) for x, y in zip(x, m_y[10:-10])])

                new_system.append(line)
            new_staff_lines.append(System(new_system))

        if self.settings.smooth_lines_adv_debug:
            f, ax = plt.subplots(1, 2, True, True)
            cmap = plt.get_cmap('jet')
            colors = cmap(np.linspace(0, 1.0, len(new_staff_lines)))
            for system, color in zip(staff_lines, colors):
                for staff in system:
                    x, y = staff.get_xy()
                    ax[0].plot(x, y, color=color)
            for system, color in zip(new_staff_lines, colors):
                for staff in system:
                    x, y = staff.get_xy()
                    ax[1].plot(x, y, color=color)
            plt.show()
        return new_staff_lines

    @staticmethod
    def best_fit_systems(system_list: List[System], gray_image: np.ndarray, binary_image: np.ndarray
                         , lt: int, scale: float = 2.0) -> List[System]:

        image_cp = gray_image  # + binary_image
        scaled_image = resize_image(image_cp, scale)

        staff_list = []
        for system in system_list:
            new_system = []
            for line in system:
                first_x_point = line.get_start_point().x
                last_x_point = line.get_end_point().x
                line = simplify_anchor_points(line, max_distance=(last_x_point - first_x_point) / 15,
                                              min_distance=(last_x_point - first_x_point) / 30)
                line.scale_line(scale)
                new_line = best_line_fit(scaled_image, line, lt, scale=scale)
                new_line.scale_line(1.0 / scale)
                new_system.append(new_line)
            staff_list.append(new_system)
        return staff_list


def remove_hill(y: List[int], smooth: int = 25) -> None:
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
                if correction:
                    break
                # correct up
                for xc in range(start + 1, start + hlen + 1):
                    yc = y[xc]
                    if not (yc < start_y and yc < end_y):
                        break
                else:
                    for xc in range(start + 1, start + hlen + 1):
                        y[xc] = min(start_y, end_y)
                        correction = True

                if correction:
                    break
            if correction:
                break


def interpolate_sequence(x_list: List[int], y_list: List[int]) -> [List[int], List[int]]:
    x_list_new = range(x_list[0], x_list[-1])
    y_list_new = np.interp(x_list_new, x_list, y_list)

    return x_list_new, list((np.floor(y_list_new + 0.5).astype(int)))


def polyline_simplification(staff_list: List[System],
                            algorithm: LineSimplificationAlgorithm = LineSimplificationAlgorithm.VISVALINGAM_WHYATT,
                            max_points_vw: int = 30, ramer_dougler_dist: float = 0.5) -> List[System]:
    new_staff_list = []
    for system in staff_list:
        new_system = []
        for line in system:
            x, y = line.get_xy()
            line_list = list(zip(x, y))
            line_array = np.asarray(line_list, dtype=np.float64)
            simplified = line_list

            if algorithm is LineSimplificationAlgorithm.VISVALINGAM_WHYATT:
                simplifier = VWSimplifier(line_array)
                simplified = simplifier.from_number(max_points_vw)
            elif algorithm is LineSimplificationAlgorithm.RAMER_DOUGLER_PEUCKLER:
                simplified = ramerdouglas(line_list, dist=ramer_dougler_dist)

            simplified = [Point(x, y) for x, y in simplified]
            new_system.append(Line(simplified))
        new_staff_list.append(System(new_system))
    return new_staff_list


def _vec2d_dist(p1: List[int], p2: List[int]):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2


def _vec2d_sub(p1: List[int], p2: List[int]):
    return p1[0]-p2[0], p1[1]-p2[1]


def _vec2d_mult(p1: List[int], p2: List[int]):
    return p1[0]*p2[0] + p1[1]*p2[1]


def check_systems(line_list: List[System], binary_image: np.ndarray, threshold: float = 0.7) -> List[System]:

    new_line_list = []
    for system in line_list:
        line_blackness = []
        for line in system:
            line_blackness.append(get_blackness_of_line(line, binary_image))
        if np.mean(line_blackness) < threshold:
            new_line_list.append(system)
    return new_line_list


def ramerdouglas(line: List[List[int]], dist: float) -> List[List[int]]:
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

    dist_sq = []
    for curr in line[1:-1]:
        tmp = (
            _vec2d_dist(begin, curr) - _vec2d_mult(_vec2d_sub(end, begin),
                                                   _vec2d_sub(curr, begin)) ** 2 / _vec2d_dist(begin, end))
        dist_sq.append(tmp)

    maxdist = max(dist_sq)
    if maxdist < dist ** 2:
        return [begin, end]

    pos = dist_sq.index(maxdist)
    return (ramerdouglas(line[:pos + 2], dist) +
            ramerdouglas(line[pos + 1:], dist)[1:])


def approximate_blackness_of_line(line: Line, image: np.ndarray) -> int:
    image = image
    x_list, y_list = line.get_xy()
    func = interpolate.interp1d(x_list, y_list)
    x_start, x_end = x_list[0], x_list[-1]
    spaced_numbers = np.linspace(x_start, x_end, num=int(abs(x_list[0] - x_list[-1]) * 1 / 5), endpoint=True)
    y_new = func(spaced_numbers)
    blackness = 0
    for ind, number in enumerate(y_new):
        if image[int(number)][int(spaced_numbers[ind])] == 255:
            blackness += 1
    return blackness


def create_data(image: np.ndarray, line_space_height: int) -> ImageData:
    norm_img = image.astype(np.float32) / 255
    binary_image = gauss_threshold(image.astype(np.uint8)) / 255
    if line_space_height == 0:
        staff_space_height, staff_line_height = vertical_runs(binary_image)
        space_height = staff_space_height + staff_line_height
    else:
        space_height = line_space_height
        staff_line_height = max(1, int(np.round(0.2 * line_space_height)))
        staff_space_height = space_height - staff_line_height

    image_data = ImageData(height=space_height, image=norm_img, staff_line_height=staff_line_height,
                           staff_space_height=staff_space_height, binary_image=binary_image)
    return image_data


if __name__ == "__main__":
    y_test = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 9, 9, 9, 9, 9, 9]
    x_test = [i for i in range(len(y_test))]
    print(y_test)
    print(x_test)
