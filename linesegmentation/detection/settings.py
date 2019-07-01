from enum import IntEnum
from typing import NamedTuple, Optional


class PostProcess(IntEnum):
    BESTFIT = 1
    FLAT = 2


class SmoothLines(IntEnum):
    OFF = 0
    BASIC = 1
    ADVANCE = 2


class OutPutType(IntEnum):
    LISTOFLISTS = 1
    LISTOFOBJECT = 2


class LineSimplificationAlgorithm(IntEnum):
    RAMER_DOUGLER_PEUCKLER = 1
    VISVALINGAM_WHYATT = 2


class LineDetectionSettings(NamedTuple):

    line_number: int = 4
    min_lines_per_system: int = 3
    horizontal_min_length: int = 6
    line_interpolation: bool = True
    debug: bool = False
    line_space_height: int = 20
    target_line_space_height: int = 10

    smooth_lines: SmoothLines = SmoothLines.OFF
    smooth_value_low_pass: float = 5
    smooth_value_adv: int = 25
    smooth_lines_adv_debug: bool = False
    line_fit_distance: float = 0.5
    model: Optional[str] = None
    model_foreground_threshold: float = 0.5
    model_foreground_normalize: bool = True

    system_threshold: float = 1.0
    debug_model: bool = False
    processes: int = 12
    post_process: PostProcess = PostProcess.BESTFIT
    post_process_debug: bool = True
    best_fit_scale: float = 2.0
    use_prediction_to_fit: bool = True
    max_line_points: int = 50

    output_type: OutPutType = OutPutType.LISTOFLISTS
