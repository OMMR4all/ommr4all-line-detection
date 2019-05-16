import numpy as np
from scipy.interpolate import interpolate
import math
from typing import List


def staff_removal(staffs_lines: List[List[List[int]]], img: np.ndarray, line_height: int):
    nimg = np.copy(img)
    h = nimg.shape[0]
    l2 = math.ceil(line_height / 2)
    l2 = max(l2, 2)
    for system in staffs_lines:
        for staff in system:
            y, x = zip(*staff)
            f = interpolate.interp1d(x, y)
            x_start, x_end = min(x), max(x)
            for i in range(x_start, x_end):
                count = []

                st_point = int(f(i))
                if nimg[st_point][i] != 0:
                    for z in range(1, l2 + 1):
                        if nimg[st_point - z][i] == 0:
                            st_point = st_point-z
                            break
                        if nimg[st_point + z][i] == 0:
                            st_point = st_point+z
                            break
                yt = st_point
                yb = st_point
                if nimg[yt][i] == 0:
                    count.append(yt)
                    while yt < h - 1:
                        yt += 1
                        if nimg[yt][i] == 0:
                            count.append(yt)
                        else:
                            break
                    while yb > 0:
                        yb -= 1
                        if nimg[yb][i] == 0:
                            count.append(yb)
                        else:
                            break
                if len(count) <= line_height:
                    for it in count:
                        nimg[it][i] = 1
    return nimg


if __name__ == "__main__":
    import os
    from linesegmentation.detection.detector import LineDetectionSettings
    from linesegmentation.detection.detection import LineDetection
    from linesegmentation.preprocessing.binarization.ocropus_binarizer import binarize
    from PIL import Image
    from linesegmentation.detection.detection import vertical_runs, calculate_horizontal_runs
    from matplotlib import pyplot as plt

    setting_predictor = LineDetectionSettings(debug=False)
    line_detector = LineDetection(setting_predictor)
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    page_path = os.path.join(project_dir, 'demo/images/Graduel_de_leglise_de_Nevers-509.nrm.png')
    for _pred in line_detector.detect_paths([page_path]):
        img = np.array(Image.open(page_path)) / 255
        binary_img = binarize(img)
        space_height_test, line_height_test = vertical_runs(binary_img)
        img_staffs_removed = staff_removal(_pred, binary_img, line_height_test)
        i, ax = plt.subplots(1, 2, True, True)
        ax[0].imshow(binary_img)
        ax[1].imshow(img_staffs_removed)
        for s in _pred:
            for l in s:
                y, x = zip(*l)
                ax[1].plot(x,y)
        plt.show()
