from ocr4all_pixel_classifier.lib.predictor import Predictor, PredictSettings
from ocr4all_pixel_classifier.lib.dataset import DatasetLoader, SingleData
from linesegmentation.datatypes.datatypes import ImageData
import numpy as np
from typing import Generator, List
from skimage.transform import resize


class PCPredictor:
    def __init__(self, settings: PredictSettings, height=20):
        self.height = height
        self.settings = settings
        self.predictor = Predictor(settings)

    def predict(self, images: List[ImageData]) -> Generator[np.array, None, None]:
        dataset_loader = DatasetLoader(self.height, prediction=True, color_map={})
        data = dataset_loader.load_data(
            [SingleData(binary=i.image * 255, image=i.image * 255, line_height_px=i.height) for i in images]
        )
        for i, pred in enumerate(self.predictor.predict(data)):
            # get the probability map for 'foreground' and resize it to the original shape
            pred = resize(pred.probabilities, pred[2].original_shape)
            yield pred

