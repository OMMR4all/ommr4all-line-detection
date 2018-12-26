import numpy as np
from scipy.interpolate import interpolate
from skimage import morphology, measure

def staff_removal(staffs_lines, img, line_height):
    nimg = np.copy(img)
    h = nimg.shape[0]
    w = nimg.shape[1]
    for system in staffs_lines:
        for staff in system:
            y, x = zip(*staff)
            f = interpolate.interp1d(x, y)
            x_start, x_end = x[0], x[-1]
            for i in range(x_start, x_end):
                count = []
                yt = int(f(i))
                yb = int(f(i))
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
    # Label all connected components:
    #all_labels = measure.label(nimg, background = 1)
    #morphology.remove_small_objects(all_labels, maxNoiseSize, in_place = True)
    #return np.clip(all_labels, 0, 1)
    return nimg

if __name__ == "__main__":
    import os
    from linesegmentation.detection.lineDetector import LineDetectionSettings
    from linesegmentation.detection.lineDetection import LineDetection
    from linesegmentation.preprocessing.binarization.ocropus_binarizer import binarize
    from PIL import Image
    from linesegmentation.detection.lineDetection import vertical_runs, calculate_horizontal_runs
    from matplotlib import pyplot as plt

    setting_predictor = LineDetectionSettings(debug=False)
    line_detector = LineDetection(setting_predictor)
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    page_path = os.path.join(project_dir, 'demo/images/Graduel_de_leglise_de_Nevers-509.nrm.png')
    for _pred in line_detector.detect([page_path]):
        img = np.array(Image.open(page_path)) / 255
        binary_img = binarize(img)
        spaceheight, lineheight = vertical_runs(binary_img)
        img_staffs_removed = staff_removal(_pred, binary_img, lineheight)
        i, ax = plt.subplots(1, 2, True, True)
        ax[0].imshow(binary_img)
        ax[1].imshow(img_staffs_removed)
        plt.show()
