# misc imports
import multiprocessing
import tqdm
from functools import partial
from typing import List, Generator

from linesegmentation.detection.datatypes import AABB
from linesegmentation.detection.detector import LineDetector, ImageData, create_data, \
    check_systems, polyline_simplification
from linesegmentation.detection.util import vertical_runs, calculate_horizontal_runs
import numpy as np
# image specific imports
from PIL import Image
from matplotlib import pyplot as plt
# project specific imports
from scipy.ndimage.morphology import binary_erosion, binary_dilation

from linesegmentation.detection.settings import LineDetectionSettings, \
    LineSimplificationAlgorithm, PostProcess, SmoothLines, OutPutType
from linesegmentation.preprocessing.binarization.ocropus_binarizer import binarize
from linesegmentation.preprocessing.enhancing.enhancer import enhance
from linesegmentation.preprocessing.util import extract_connected_components, \
    normalize_connected_components
from linesegmentation.detection.callback import LineDetectionCallback, DummyLineDetectionCallback
from linesegmentation.detection.detector import System, Line, Point


class LineDetection(LineDetector):
    """Line detection class

    Attributes
    ----------
    settings : LineDetectionSettings
        Setting for the line detection algorithm
    """

    def __init__(self, settings: LineDetectionSettings, callback: LineDetectionCallback = None):
        """Constructor of the LineDetection class

        Parameters
        ----------
        settings: LineDetectionSettings
            Settings for the line detection algorithm
        """
        super().__init__(settings)
        if callback is None:
            self.callback: LineDetectionCallback = DummyLineDetectionCallback()
        else:
            self.callback: LineDetectionCallback = callback

    def detect_paths(self, image_paths: List[str]) -> Generator[List[List[List[int]]], None, None]:
        def read_img(path):
            return np.array(Image.open(path))

        return self.detect(list(map(read_img, image_paths)))

    def detect(self, images: List[np.ndarray]) -> Generator[List[List[List[List[int]]]], None, None]:
        """
        Function  to detect die stafflines in an image

        Parameters
        ----------
        images: List[np.ndarray]
            Raw gray scale image in range [0, 255], which should be processed

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
            return self.detect_morphological(images)
        else:
            return self.detect_fcn(images)

    def detect_morphological(self, images: List[np.ndarray]) -> Generator[List[List[List[List[int]]]], None, None]:

        for img in images:

            image_data = ImageData()
            image_data.image = img.astype(float) / 255
            gray = image_data.image.copy()

            if np.sum(np.histogram(gray)[0][1:-2]) != 0:
                gray = enhance(image_data.image)

            binary = binarize(gray)
            binarized = 1 - binary
            morph = binary_erosion(binarized, structure=np.full((5, 1), 1))
            morph = binary_dilation(morph, structure=np.full((5, 1), 1))
            staffs = (binarized ^ morph)
            image_data.staff_space_height, image_data.staff_line_height = vertical_runs(binary)
            image_data.binary_image = binary
            self.callback.update_total_state()
            image_data.horizontal_runs_img = calculate_horizontal_runs((1 - staffs),
                                                                       self.settings.horizontal_min_length)
            self.callback.update_total_state()
            yield self.detect_staff_lines(image_data)
            self.callback.update_total_state()
            self.callback.update_page_counter()

    def detect_prob_map(self, image: np.ndarray, probability_map) -> List[List[List[List[int]]]]:
        data = create_data(image, line_space_height=self.settings.line_space_height)
        prob = probability_map
        if self.settings.model_use_argmax:
            pred = np.argmax(prob, axis=-1)
        else:
            prob = prob[:, :, 1]
            norm = prob / np.max(prob) if self.settings.model_foreground_normalize else prob
            pred = (norm > self.settings.model_foreground_threshold)
        if data.staff_space_height is None or data.staff_line_height is None:
            data.staff_space_height, data.staff_line_height = vertical_runs(data.binary_image)
        data.horizontal_runs_img = calculate_horizontal_runs(1 - pred, self.settings.horizontal_min_length)
        data.pixel_classifier_prediction = prob

        if self.settings.debug_model:
            f, ax = plt.subplots(2, 3, sharex='all', sharey='all')
            ax[0, 0].imshow(1 - prob, cmap='gray')
            ax[0, 1].imshow(1 - pred, cmap='gray')
            ax[0, 2].imshow(255 - data.horizontal_runs_img, cmap='gray')
            ax[1, 0].imshow(data.image, cmap='gray')
            plt.show()
        return self.detect_staff_lines(data)
    def detect_fcn(self, images: List[np.ndarray]) -> Generator[List[List[List[List[int]]]], None, None]:
        self.callback.set_total_pages(len(images))
        create_data_partial = partial(create_data, line_space_height=self.settings.line_space_height)
        if len(images) <= 1:
            data = list(map(create_data_partial, images))
        else:
            with multiprocessing.Pool(processes=self.settings.processes) as p:
                data = [v for v in tqdm.tqdm(p.imap(create_data_partial, images), total=len(images))]
        for i, prob in enumerate(self.predictor.predict(data)):
            prob = prob
            if self.settings.model_use_argmax:
                pred = np.argmax(prob, axis=-1)
            else:
                prob = prob[:, :, 1]
                norm = prob / np.max(prob) if self.settings.model_foreground_normalize else prob
                pred = (norm > self.settings.model_foreground_threshold)
            self.callback.update_total_state()
            if data[i].staff_space_height is None or data[i].staff_line_height is None:
                data[i].staff_space_height, data[i].staff_line_height = vertical_runs(data[i].binary_image)
            data[i].horizontal_runs_img = calculate_horizontal_runs(1 - pred, self.settings.horizontal_min_length)
            data[i].pixel_classifier_prediction = prob

            self.callback.update_total_state()
            if self.settings.debug_model:
                f, ax = plt.subplots(2, 3, sharex='all', sharey='all')
                ax[0, 0].imshow(1-prob, cmap='gray')
                ax[0, 1].imshow(1 - pred, cmap='gray')
                ax[0, 2].imshow(255 - data[i].horizontal_runs_img, cmap='gray')
                ax[1, 0].imshow(data[i].image, cmap='gray')
                plt.show()
            yield self.detect_staff_lines(data[i])
            self.callback.update_total_state()
            self.callback.update_page_counter()
            

    def detect_staff_lines(self, image_data: ImageData) -> List[List[List[List[int]]]]:
        img = image_data.horizontal_runs_img

        binary_image = image_data.binary_image
        staff_line_height = image_data.staff_line_height
        staff_space_height = image_data.staff_space_height

        def compute_aabb_with_bounds(line: Line) -> AABB:
            aabb = line.aabb().copy()
            aabb.expand(((staff_space_height + staff_line_height) * 2, 0))
            return aabb

        def remove_abandoned_ccs(ccs: List[Line]) -> List[Line]:
            aabbs = list(map(compute_aabb_with_bounds, ccs))
            return [cc for cc, aabb in zip(ccs, aabbs) if any(aabb.intersects(aabb2) for aabb2 in aabbs)]

        def group_ccs_into_cols(ccs: List[Line]) -> List[List[Line]]:
            columns = []
            for cc in ccs:
                left = cc.get_start_point().x
                right = cc.get_end_point().x
                matched_columns = [c for c in columns if c['left'] <= right and c['right'] >= left]
                columns.append({
                    'lines': sum([c['lines'] for c in matched_columns], []) + [cc],
                    'left': min([c['left'] for c in matched_columns] + [left]),
                    'right': max([c['right'] for c in matched_columns] + [right]),
                })
                for mc in matched_columns:
                    columns.remove(mc)

            return [c['lines'] for c in columns]

        cc_list = self.extract_ccs(img)
        cc_list = remove_abandoned_ccs(cc_list)
        columns = group_ccs_into_cols(cc_list)

        columns_line_list = [self.connect_connected_components_to_line(lines[:], staff_line_height, staff_space_height) for lines in columns]

        if self.settings.debug:
            f, ax = plt.subplots(1, 3)
            ax[0].imshow(np.zeros(image_data.image.shape, dtype=np.uint8) + 255, cmap='gray', vmin=0, vmax=1)
            for line in cc_list:
                x, y = Line(line).get_xy()
                ax[0].plot(x, y, "k")

            ax[1].imshow(np.zeros(image_data.image.shape, dtype=np.uint8) + 255, cmap='gray', vmin=0, vmax=1)
            for line in sum(columns_line_list, []):
                x, y = line.get_xy()
                ax[1].plot(x, y, "k")

            ax[2].imshow(255 - img, cmap='gray')
            plt.show()

        self.callback.update_total_state()

        def prune_and_normalize(line_list: List[Line]):
            line_list = self.prune_small_lines(line_list, staff_space_height)
            if self.settings.line_number > 1:
                staff_list = self.organize_lines_in_systems(line_list, staff_space_height, staff_line_height)

                staff_list = self.prune_lines_in_system_with_lowest_intensity(staff_list, img)
                if self.settings.line_interpolation:
                    staff_list = self.normalize_lines_in_system(staff_list, staff_space_height, img)

            else:
                staff_list = [[System(x)] for x in line_list]

            return staff_list

        columns_staff_list = list(map(prune_and_normalize, columns_line_list))

        self.callback.update_total_state()

        def post_process_staves(staff_list):
            if self.settings.post_process == PostProcess.FLAT:
                staff_list = self.post_process_staff_systems(staff_list, staff_line_height, binary_image)
                if self.settings.line_number > 1 and self.settings.line_interpolation:
                    staff_list = self.normalize_lines_in_system(staff_list, staff_space_height, img)
                self.callback.update_total_state()
                if self.settings.smooth_lines != SmoothLines.OFF:
                    if self.settings.smooth_lines == SmoothLines.BASIC:
                        staff_list = self.smooth_lines(staff_list)
                    if self.settings.smooth_lines == SmoothLines.ADVANCE:
                        staff_list = self.smooth_lines_advanced(staff_list)

                    if self.settings.line_fit_distance > 0:
                        staff_list = polyline_simplification(staff_list,
                                                             algorithm=LineSimplificationAlgorithm.RAMER_DOUGLER_PEUCKLER,
                                                             ramer_dougler_dist=self.settings.line_fit_distance)
                self.callback.update_total_state()

            elif self.settings.post_process == PostProcess.BESTFIT:

                staff_list = polyline_simplification(staff_list, algorithm=LineSimplificationAlgorithm.VISVALINGAM_WHYATT,
                                                     max_points_vw=self.settings.max_line_points)
                self.callback.update_total_state()

                staff_list = self.best_fit_systems(staff_list,
                                                   1 - image_data.pixel_classifier_prediction
                                                   if image_data.pixel_classifier_prediction is not None and
                                                      self.settings.use_prediction_to_fit
                                                   else image_data.image,
                                                   image_data.binary_image, staff_line_height,
                                                   self.settings.best_fit_scale)
                self.callback.update_total_state()

                staff_list = polyline_simplification(staff_list,
                                                     algorithm=LineSimplificationAlgorithm.RAMER_DOUGLER_PEUCKLER,
                                                     ramer_dougler_dist=0.5)

            return staff_list

        #staff_list = check_systems(staff_list, binary_image, threshold=self.settings.line_fit_distance)
        columns_staff_list = [post_process_staves(sl) for sl in columns_staff_list]
        staff_list = sum(columns_staff_list, [])

        self.callback.update_total_state()
        # Debug
        if self.settings.debug:
            extent_b = (0, image_data.image.shape[1], image_data.image.shape[0], 0)

            plt.imshow(image_data.image, cmap='gray', extent=extent_b)
            cmap = plt.get_cmap('jet')
            colors = cmap(np.linspace(0, 1.0, len(staff_list)))

            for system, color in zip(staff_list, colors):
                for staff in system:
                    x, y = staff.get_xy()
                    plt.plot(x, y, color=color)
                    plt.plot(x, y, "bo")

            plt.show()

        if self.settings.output_type == OutPutType.LISTOFLISTS:
            staff_list_result = []
            for system in staff_list:
                new_system = []
                for line in system:
                    x, y = line.get_xy()
                    new_line = list(zip(y, x))
                    new_system.append(new_line)
                staff_list_result.append(new_system)
            return staff_list_result
        else:
            return staff_list


if __name__ == "__main__":
    import os
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    model_line = os.path.join(project_dir, '/home/alexanderh/models/datset/model.h5')
    setting_predictor = LineDetectionSettings(debug=True, line_space_height=10,post_process=1, model=model_line, debug_model=True)

    t_callback = DummyLineDetectionCallback(total_steps=8, total_pages=2)
    line_detector = LineDetection(setting_predictor, t_callback)

    page_path = os.path.join(project_dir, 'demo/images/Graduel_de_leglise_de_Nevers-509.nrm.png')
    for _pred in line_detector.detect_paths(['/tmp/35.png']):
        pass
