# misc imports
import multiprocessing
import tqdm
from functools import partial
from typing import List, Generator
from linesegmentation.detection.lineDetectionUtil import vertical_runs, calculate_horizontal_runs
import numpy as np
# image specific imports
from PIL import Image
from matplotlib import pyplot as plt
# project specific imports
from scipy.ndimage.morphology import binary_erosion, binary_dilation
from linesegmentation.detection.lineDetector import LineDetector, LineDetectionSettings, ImageData, create_data
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
    setting_predictor = LineDetectionSettings(debug=True)
    line_detector = LineDetection(setting_predictor)
    for _pred in line_detector.detect(['/home/alexanderh/Schreibtisch/masterarbeit/OMR/Graduel_de_leglise_de_Nevers/interesting/part1/bin/Graduel_de_leglise_de_Nevers-023.nrm.png']):
        pass
