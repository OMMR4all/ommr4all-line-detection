import multiprocessing
import tqdm
from PIL import Image
import numpy as np
from functools import partial
from matplotlib import pyplot as plt
from linesegmentation.detection.detector import LineDetector, LineDetectionSettings, create_data, ImageData,\
    LineSimplificationAlgorithm, polyline_simplification
from linesegmentation.detection.util import get_text_borders, vertical_runs, calculate_horizontal_runs
from pagesegmentation.lib.predictor import PredictSettings
from linesegmentation.pixelclassifier.predictor import PCPredictor
from linesegmentation.preprocessing.binarization.ocropus_binarizer import binarize
from linesegmentation.preprocessing.util import extract_connected_components, \
    normalize_connected_components
from typing import List, Generator
from linesegmentation.detection.datatypes import Line, System, Point
from linesegmentation.detection.callback import DummyLineDetectionCallback, LineDetectionCallback


class LineDetectionRest(LineDetector):
    '''
    Line detection for images with virtual staff lines
    '''
    def __init__(self, settings, callback: LineDetectionCallback = None, text_model=None):
        super().__init__(settings)
        self.text_predictor = None
        if text_model:
            pc_settings_text = PredictSettings(
                mode='meta',
                network=os.path.abspath(text_model),
                output=None,
                high_res_output=False
            )
            self.text_predictor = PCPredictor(pc_settings_text, settings.target_line_space_height)
        if callback is None:
            self.callback: LineDetectionCallback = DummyLineDetectionCallback()
        else:
            self.callback: LineDetectionCallback = callback

    def detect_paths(self, image_paths: List[str]) -> Generator[List[List[List[int]]], None, None]:
        def read_img(path):
            return np.array(Image.open(path))

        return self.detect_advanced(list(map(read_img, image_paths)))

    def detect_advanced(self, images: List[np.ndarray]):
        create_data_partial = partial(create_data, line_space_height=self.settings.line_space_height)
        with multiprocessing.Pool(processes=self.settings.processes) as p:
            data = [v for v in tqdm.tqdm(p.imap(create_data_partial, images), total=len(images))]

        if self.text_predictor:
            for i, pred in enumerate(zip(self.predictor.predict(data), self.text_predictor.predict(data))):
                self.callback.reset_page_state()

                line_prediction = pred[0]
                region_prediction = pred[1]
                t_region = np.clip(pred[1], 0, 1) * 255
                line_prediction[line_prediction > 0] = 255
                region_prediction[region_prediction < 255] = 0
                line_prediction_pruned = line_prediction * region_prediction * 255
                self.callback.update_current_page_state()

                if self.settings.debug:
                    x, y = plt.subplots(1, 3, sharex=True, sharey=True)
                    y[0].imshow(region_prediction)
                    y[1].imshow(line_prediction)
                    y[2].imshow(line_prediction_pruned)
                    plt.show()
                self.callback.update_current_page_state()

                data[i].staff_space_height, data[i].staff_line_height = vertical_runs(1 - line_prediction_pruned)
                self.callback.update_current_page_state()

                data[i].horizontal_runs_img = calculate_horizontal_runs((1 - (line_prediction_pruned / 255)),
                                                                        self.settings.horizontal_min_length)
                self.callback.update_current_page_state()
                yield self.detect_staff_lines_rest(data[i], get_text_borders(t_region - region_prediction))
                self.callback.update_total_state()

        else:
            for i, pred in enumerate(self.predictor.predict(data)):
                self.callback.reset_page_state()

                pred[pred > 0] = 255
                self.callback.update_current_page_state()

                data[i].staff_space_height, data[i].staff_line_height = vertical_runs(1 - pred)
                self.callback.update_current_page_state()

                data[i].horizontal_runs_img = calculate_horizontal_runs((1 - (pred / 255)),
                                                                        self.settings.horizontal_min_length)
                self.callback.update_current_page_state()

                data[i].image = np.array(Image.open(data[i].path)) / 255
                binary = np.array(binarize(data[i].image), dtype='uint8')
                text_borders = get_text_borders((1 - binary)*255, preprocess=True)
                self.callback.update_current_page_state()

                yield self.detect_staff_lines_rest(data[i], text_borders)
                self.callback.update_total_state()

    def organize_lines_in_systems(self, line_list: List[System], staff_space_height: int,
                                  staff_line_height: int, text_height: int):
        mean_line_height_list = [line.get_average_line_height() for line in line_list]
        staffindices = []
        prev_text_height = 0
        for th in text_height:
            m_staff_height_between_interval = [x for x in mean_line_height_list if prev_text_height <= x <= th]
            for i, medium_y in enumerate(mean_line_height_list):
                system = []
                if i in sum(staffindices, []) or medium_y not in m_staff_height_between_interval:
                    continue
                height = medium_y
                for z, center_ys in enumerate(mean_line_height_list):
                    if np.abs(height - center_ys) < 2.1 * (
                            staff_space_height + staff_line_height) and center_ys in m_staff_height_between_interval:
                        system.append(z)
                        height = center_ys
                staffindices.append(system)
            prev_text_height = th
        staffindices = [staff for staff in staffindices if len(staff) >= self.settings.min_lines_per_system]
        staff_list = []
        for z in staffindices:
            system_list = []
            for x in z:
                system.append(line_list[x])
            staff_list.append(System(system_list))
        return staff_list

    def detect_staff_lines_rest(self, image_data: ImageData, text_height: int):
        img = image_data.horizontal_runs_img
        staff_line_height = image_data.staff_line_height
        staff_space_height = image_data.staff_space_height

        cc_list = self.extract_ccs(img)
        self.callback.update_current_page_state()

        line_list = self.connect_connected_components_to_line(cc_list, staff_line_height, staff_space_height)
        self.callback.update_current_page_state()

        # Remove lines which are shorter than 50px
        line_list = [l for l in line_list if l[-1][1] - l[0][1] > 50]

        line_list = self.prune_small_lines(line_list, staff_space_height)
        self.callback.update_current_page_state()

        if self.settings.line_number > 1:
            staff_list = self.organize_lines_in_systems(line_list, staff_space_height, staff_line_height, text_height)
            staff_list = self.prune_lines_in_system_with_lowest_intensity(staff_list, img)
            if self.settings.line_interpolation:
                staff_list = self.normalize_lines_in_system(staff_list, staff_space_height, img)

        else:
            staff_list = [[x] for x in line_list]
        self.callback.update_current_page_state()

        if self.settings.smooth_lines != 0:
            if self.settings.smooth_lines == 1:
                staff_list = self.smooth_lines(staff_list)
            if self.settings.smooth_lines == 2:
                staff_list = self.smooth_lines_advanced(staff_list)
        self.callback.update_current_page_state()

        staff_list = polyline_simplification(staff_list, algorithm=LineSimplificationAlgorithm.RAMER_DOUGLER_PEUCKLER,
                                             ramer_dougler_dist=self.settings.line_fit_distance)
        self.callback.update_current_page_state()

        # Debug
        if self.settings.debug:
            f, ax = plt.subplots(1, 2, sharex=True, sharey=True)
            ax[0].imshow(image_data.image, cmap='gray')
            cmap = plt.get_cmap('jet')
            colors = cmap(np.linspace(0, 1.0, len(staff_list)))
            for system, color in zip(staff_list, colors):
                for staff in system:
                    x, y = staff.get_xy()
                    ax[0].plot(x, y, color=color)
            ax[1].imshow(img, cmap='gray')
            plt.show()
        return staff_list


if __name__ == "__main__":
    import os

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    page_path = os.path.join(project_dir, 'demo/images/Graduel_de_leglise_de_Nevers-427.nrm.png')
    model_line = os.path.join(project_dir, 'demo/models/line/virtual_lines/model')
    model_region = os.path.join(project_dir, 'demo/models/region/model')
    settings_prediction = LineDetectionSettings(debug=True, min_lines_per_system=3, line_number=4, line_space_height=20,
                                                target_line_space_height=10, smooth_lines=2, line_fit_distance=1.0,
                                                model=model_line)
    t_callback = DummyLineDetectionCallback(total_steps=10, total_pages=1)
    line_detector = LineDetectionRest(settings_prediction, text_model=model_region, callback=t_callback)

    for _pred in line_detector.detect_paths([page_path]):
        pass
