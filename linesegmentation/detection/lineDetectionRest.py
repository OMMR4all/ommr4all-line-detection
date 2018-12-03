import os
import multiprocessing
import tqdm
from functools import partial
from matplotlib import pyplot as plt
from linesegmentation.detection.lineDetection import LineDetection, create_data, vertical_runs,\
    calculate_horizontal_runs, LineDetectionSettings
from pagesegmentation.lib.predictor import PredictSettings
from linesegmentation.pixelclassifier.predictor import PCPredictor
from typing import List, Generator

class lineDetectionRest():
    def __init__(self, settings, text_model=None):
        self.settings = settings
        self.predictor = None
        if settings.model:
            pc_settings = PredictSettings(
                mode='meta',
                network=os.path.abspath(settings.model),
                output=None,
                high_res_output=False
            )
            self.predictor = PCPredictor(pc_settings, settings.targetLineSpaceHeight)
        self.text_predictor = None
        if text_model:
            pc_settings_text = PredictSettings(
                mode='meta',
                network=os.path.abspath(text_model),
                output=None,
                high_res_output=False
            )
            self.text_predictor = PCPredictor(pc_settings_text, settings.targetLineSpaceHeight)

        self.line_detector = LineDetection(settings)

    def detect_advanced(self, image_paths: List[str]):
        create_data_partial = partial(create_data, line_space_height=self.settings.lineSpaceHeight)
        with multiprocessing.Pool(processes=self.settings.processes) as p:
            data = [v for v in tqdm.tqdm(p.imap(create_data_partial, image_paths), total=len(image_paths))]
        if self.text_predictor:
            for i, pred in enumerate(zip(self.predictor.predict(data), self.text_predictor.predict(data))):
                line_prediction = pred[0]
                region_prediction = pred[1]
                # test
                line_prediction[line_prediction > 0] = 255
                region_prediction[region_prediction < 255] = 0
                line_prediction_pruned = line_prediction * region_prediction * 255
                if self.settings.debug:
                    x, y = plt.subplots(1, 3, True, True)
                    y[0].imshow(region_prediction)
                    y[1].imshow(line_prediction)
                    y[2].imshow(line_prediction_pruned)
                    plt.show()
                data[i].staff_space_height, data[i].staff_line_height = vertical_runs(1 - line_prediction_pruned)
                data[i].horizontal_runs_img = calculate_horizontal_runs((1 - (line_prediction_pruned / 255)),
                                                                        self.settings.minLength)
                yield self.line_detector.detect_staff_lines((data[i]))
        else:
             for i, pred in enumerate(self.predictor.predict(data)):
                 pred[pred > 0] = 255
                 data[i].staff_space_height, data[i].staff_line_height = vertical_runs(1 - pred)
                 data[i].horizontal_runs_img = calculate_horizontal_runs((1 - (pred / 255)),
                                                                         self.settings.minLength)
                 yield self.line_detector.detect_staff_lines((data[i]))

if __name__ == "__main__":
    settings = LineDetectionSettings(debug=True, minLineNum=2, numLine=6, lineSpaceHeight=20,targetLineSpaceHeight=10,
                                     model='/home/alexanderh/Schreibtisch/masterarbeit/models/line/restnrmthl10')
    line_detector = lineDetectionRest(settings,'/home/alexanderh/Schreibtisch/masterarbeit/models/region//model')
    for pred in line_detector.detect_advanced(
            ['/home/alexanderh/Schreibtisch/masterarbeit/OMR/Graduel_de_leglise_de_Nevers/restnrm/Graduel_de_leglise_de_Nevers-429.nrm.png']):
        pass