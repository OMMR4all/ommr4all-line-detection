import numpy as np
from PIL import Image, ImageStat
import math

def normalize_raw_image(raw):
    image = raw.astype(np.float32) - np.amin(raw)
    image /= np.amax(raw)
    return image


def smooth_array(values, smoothing):
    value = values[0] # start with the first input
    for i in range(len(values)):
        current_value = values[i]
        value += (current_value - value) / smoothing
        values[i] = int(math.floor(value + 0.5))
    return values


def detect_color_image(file, thumb_size=40, MSE_cutoff=22, adjust_color_bias=True):
    pil_img = Image.open(file)
    bands = pil_img.getbands()
    if bands == ('R','G','B') or bands== ('R','G','B','A'):
        thumb = pil_img.resize((thumb_size,thumb_size))
        SSE, bias = 0, [0,0,0]
        if adjust_color_bias:
            bias = ImageStat.Stat(thumb).mean[:3]
            bias = [b - sum(bias)/3 for b in bias ]
        for pixel in thumb.getdata():
            mu = sum(pixel)/3
            SSE += sum((pixel[i] - mu - bias[i])*(pixel[i] - mu - bias[i]) for i in [0,1,2])
        MSE = float(SSE)/(thumb_size*thumb_size)
        if MSE <= MSE_cutoff:
            return "grayscale"            
        else:
            return "color"            
        print("( MSE=",MSE,")")
    elif len(bands)==1:
        return "binary"
    else:
       return "other"

if __name__ == "__main__":
    import os
    from linesegmentation.detection.lineDetector import LineDetectionSettings
    from linesegmentation.detection.lineDetection import LineDetection
    from PIL import Image
    from matplotlib import pyplot as plt

    setting_predictor = LineDetectionSettings(debug=False, post_process=True)
    line_detector = LineDetection(setting_predictor)
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    page_path = os.path.join(project_dir, 'demo/images/Graduel_de_leglise_de_Nevers-509.nrm.png')
    for _pred in line_detector.detect([page_path]):
        im = plt.imread(page_path)
        f, ax = plt.subplots(1, 2, True, True)
        ax[0].imshow(im, cmap='gray')
        cmap = plt.get_cmap('jet')
        colors = cmap(np.linspace(0, 1.0, len(_pred)))
        for system, color in zip(_pred, colors):
            for staff in system:
                y, x = zip(*staff)
                ax[0].plot(x, y, color=color)
        ax[1].imshow(im, cmap='gray')
        for system, color in zip(_pred, colors):
            for staff in system:
                y, x = zip(*staff)
                y = smooth_array(list(y), 1.3)
                ax[1].plot(x, y, color=color)
        plt.show()
