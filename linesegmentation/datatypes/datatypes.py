from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class ImageData:
    path: str = None
    height: int = None
    image: np.array = None
    horizontal_runs_img: np.array = None
    staff_line_height: int = None
    staff_space_height: int = None
    binary_image: np.array = None
    pixel_classifier_prediction: np.array = None
