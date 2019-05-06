# misc imports
import multiprocessing
import tqdm
from functools import partial
from typing import List, Generator
from linesegmentation.detection.lineDetectionUtil import vertical_runs, calculate_horizontal_runs, get_blackness_of_line
import numpy as np
# image specific imports
from PIL import Image
from matplotlib import pyplot as plt
# project specific imports
from scipy.ndimage.morphology import binary_erosion, binary_dilation
from linesegmentation.detection.lineDetector import LineDetector, LineDetectionSettings, ImageData, create_data, \
    line_fitting, check_systems
from linesegmentation.preprocessing.binarization.ocropus_binarizer import binarize
from linesegmentation.preprocessing.enhancing.enhancer import enhance
from linesegmentation.preprocessing.preprocessingUtil import extract_connected_components, normalize_connected_components

class LineDetection(LineDetector):
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
        super().__init__(settings)

    def detect_paths(self, image_paths: List[str]) -> Generator[List[List[List[int]]], None, None]:
        def read_img(path):
            return np.array(Image.open(path))

        return self.detect(list(map(read_img, image_paths)), image_paths)

    def detect(self, images: List[np.ndarray], image_paths) -> Generator[List[List[List[int]]], None, None]:
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
            return self.detect_basic(images)
        else:
            return self.detect_advanced(images)

    def detect_basic(self, images: List[np.ndarray]) -> Generator[List[List[List[int]]], None, None]:

        for img in images:
            image_data = ImageData()
            image_data.image = img.astype(float) / 255
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

    def detect_advanced(self, images: List[np.ndarray]) -> Generator[List[List[List[int]]], None, None]:
        create_data_partital = partial(create_data, line_space_height=self.settings.lineSpaceHeight)
        with multiprocessing.Pool(processes=self.settings.processes) as p:
            data = [v for v in tqdm.tqdm(p.imap(create_data_partital, images), total=len(images))]
        for i, prob in enumerate(self.predictor.predict(data)):
            pred = (prob > self.settings.model_foreground_threshold)

            if data[i].staff_space_height is None or data[i].staff_line_height is None:
                data[i].staff_space_height, data[i].staff_line_height = vertical_runs(data[i].binary_image)
            data[i].horizontal_runs_img = calculate_horizontal_runs(1 - pred, self.settings.minLength)

            if self.settings.debug_model:
                f, ax = plt.subplots(1, 3, sharex='all', sharey='all')
                ax[0].imshow(prob)
                ax[1].imshow(pred)
                ax[2].imshow(data[i].horizontal_runs_img)
                plt.show()
            yield self.detect_staff_lines(data[i])

    def detect_staff_lines(self, image_data: ImageData):
        img = image_data.horizontal_runs_img
        binary_image = image_data.binary_image
        staff_line_height = image_data.staff_line_height
        staff_space_height = image_data.staff_space_height

        cc_list = extract_connected_components(img)

        cc_list = normalize_connected_components(cc_list)

        line_list = self.connect_connected_components_to_line(cc_list, staff_line_height, staff_space_height)

        # Remove lines which are shorter than 50px
        line_list = [l for l in line_list if l[-1][1] - l[0][1] > 50]

        # Debug
        staff2 = line_list.copy()

        line_list = self.prune_small_lines(line_list, staff_space_height)

        if self.settings.numLine > 1:
            staff_list = self.organize_lines_in_systems(line_list, staff_space_height, staff_line_height)

            staff_list = self.prune_lines_in_system_with_lowest_intensity(staff_list, img)

            if self.settings.lineExtension:
                staff_list = self.normalize_lines_in_system(staff_list, staff_space_height, img)

        else:
            staff_list = [[x] for x in line_list]
        stafflist2 = line_fitting(staff_list, 1)

        if self.settings.post_process:
            staff_list = self.postprocess_staff_systems(staff_list, staff_line_height, binary_image)
            if self.settings.numLine > 1 and self.settings.lineExtension:
                staff_list = self.normalize_lines_in_system(staff_list, staff_space_height, img)

            if self.settings.smooth_lines != 0:
                if self.settings.smooth_lines == 1:
                    staff_list = self.smooth_lines(staff_list)
                if self.settings.smooth_lines == 2:
                    staff_list = self.smooth_lines_advanced(staff_list)

                if self.settings.line_fit_distance > 0:
                    staff_list = line_fitting(staff_list, self.settings.line_fit_distance)

        elif True:
            staff_list = line_fitting(staff_list, 1)
            staff_list = self.best_fit_systems(staff_list, image_data.image, staff_line_height)

        staff_list = check_systems(staff_list, binary_image)

        # Debug
        if self.settings.debug:
            f, ax = plt.subplots(1, 2, True, True)
            ax[0].imshow(binary_image, cmap='gray')
            cmap = plt.get_cmap('jet')
            colors = cmap(np.linspace(0, 1.0, len(staff_list)))
            for system, color in zip(stafflist2, colors):
                for staff in system:
                    y, x = zip(*staff)
                    ax[0].plot(x, y, color=color)
                    ax[0].plot(x, y, "bo")

            ax[1].imshow(image_data.image, cmap='gray')
            for system, color in zip(staff_list, colors):
                for staff in system:
                    y, x = zip(*staff)
                    ax[1].plot(x, y, color=color)
                    ax[1].plot(x, y, "bo")
            plt.show()
        return staff_list


if __name__ == "__main__":
    import os
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_line = os.path.join(project_dir, 'demo/models/line/marked_lines/best')
    setting_predictor = LineDetectionSettings( debug=True,
                                              model=model_line)
    line_detector = LineDetection(setting_predictor)

    page_path = os.path.join(project_dir, 'demo/images/Graduel_de_leglise_de_Nevers-509.nrm.png')

    for _pred in line_detector.detect_paths([page_path]):
        pass
