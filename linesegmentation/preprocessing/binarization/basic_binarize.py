from skimage.morphology import disk, square
from skimage.filters import threshold_otsu, threshold_adaptive
from skimage.filters.rank import otsu, threshold
import cv2 as cv
from datetime import datetime


def binarize_global(image):
    global_thresh = threshold_otsu(image)
    binary_global = image > global_thresh
    return binary_global


# deprecated
#def binarize_adaptive(image, block_size=35, offset = 40):
#    binary_adaptive = threshold_adaptive(image=image, block_size = block_size, offset=offset)
#    return binary_adaptive


def adaptive_otsu(image, radius = 55):
    selem = disk(radius)
    #print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
    local_otsu = otsu(image, selem)
    #print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
    binary = image >= local_otsu
    #print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
    return binary


def gauss_threshold(image, block_size=35, offset = 40):
    binary = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, \
                               cv.THRESH_BINARY, block_size, offset)
    return binary


if __name__ == "__main__":
    from PIL import Image
    import numpy as np
    import os
    from matplotlib import pyplot as plt

    def read_img(path):
        return np.array(Image.open(path))

    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    page_path = os.path.join(project_dir, 'demo/images/Graduel_de_leglise_de_Nevers-509.nrm.png')

    image_to_binarize = read_img(page_path)
    plt.imshow(gauss_threshold(image_to_binarize))
    plt.show()
