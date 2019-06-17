import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(BASE_DIR))
import unittest
import logging
from linesegmentation.detection.detection import LineDetection
from linesegmentation.detection.settings import LineDetectionSettings
from linesegmentation.detection.callback import DummyLineDetectionCallback

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', stream=sys.stdout)


# Change database to test storage

class TestStringMethods(unittest.TestCase):

    def setUp(self) -> None:
        print(BASE_DIR)

        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_line = os.path.join(self.project_dir, 'demo/models/line/marked_lines/model')
        self.t_callback = DummyLineDetectionCallback(total_steps=7, total_pages=1)

    def test_line_001_no_model(self):
        page_path = os.path.join(self.project_dir, 'demo/images/Graduel_de_leglise_de_Nevers-509.nrm.png')

        setting_predictor = LineDetectionSettings(debug=False, post_process=1)  # , model=model_line)
        self.line_detection(page_path, setting_predictor, self.t_callback, 12)

    def test_line_001_model(self):
        page_path = os.path.join(self.project_dir, 'demo/images/Graduel_de_leglise_de_Nevers-509.nrm.png')

        setting_predictor = LineDetectionSettings(debug=False, post_process=1, model=self.model_line)
        self.line_detection(page_path, setting_predictor, self.t_callback, 12)

    def test_line_002_no__model(self):
        page_path = os.path.join(self.project_dir, 'demo/images/Graduel_de_leglise_de_Nevers-003.nrm.jpg')

        setting_predictor = LineDetectionSettings(debug=False, post_process=1)  # , model=model_line)
        self.line_detection(page_path, setting_predictor, self.t_callback, 9)

    def test_line_002_model(self):
        page_path = os.path.join(self.project_dir, 'demo/images/Graduel_de_leglise_de_Nevers-003.nrm.jpg')

        setting_predictor = LineDetectionSettings(debug=False, post_process=1, model= self.model_line)
        self.line_detection(page_path, setting_predictor, self.t_callback, 9)

    def line_detection(self, page, setting, callback, number_of_systems):
        line_detector = LineDetection(setting, callback)
        for _pred in line_detector.detect_paths([page]):
            self.assertEqual(len(_pred), number_of_systems)


if __name__ == '__main__':
    unittest.main()
