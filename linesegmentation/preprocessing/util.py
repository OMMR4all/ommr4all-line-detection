import cv2
import operator
from collections import defaultdict
import numpy as np
from PIL import Image
from skimage.transform import rescale
from typing import List


def extract_connected_components(image: np.ndarray):
    connectivity = 8
    output = cv2.connectedComponentsWithStats(image, connectivity)
    ccdict = defaultdict(list)
    indexdim0, indexdim1 = np.array(output[1]).nonzero()
    points = list(zip(indexdim0, indexdim1))
    for p in points:
        y_coord, x_coord = p[0], p[1]
        k = output[1][y_coord][x_coord]
        ccdict[k].append([y_coord, x_coord])
    cc_list = list(ccdict.values())
    [x.sort(key=operator.itemgetter(1)) for x in cc_list]
    return cc_list


def normalize_connected_components(cc_list: List[List[int]]):
    # Normalize the CCs (line segments), so that the height of each cc is normalized to one pixel
    def normalize(point_list):
        normalized_cc_list = []
        for cc in point_list:
            cc_dict = defaultdict(list)
            for y, x in cc:
                cc_dict[x].append(y)
            normalized_cc = []
            for key, value in cc_dict.items():
                normalized_cc.append([int(np.floor(np.mean(value) + 0.5)), key])
            normalized_cc_list.append(normalized_cc)
        return normalized_cc_list
    return normalize(cc_list)


def resize_image(image: np.ndarray, scale: float, order=3):
    return rescale(image, scale, preserve_range=True, order=order)


def convert_2dpoint_to_1did(list: List[int], width: int):
    point_to_id = list[1] * width + list[0]
    return point_to_id


def convert_2darray_to_1darray(array: np.ndarray, width: int):
    return array[:, 0] * width + array[:, 1]


if __name__ == "__main__":
    import os
    from matplotlib import pyplot as plt
    #l = np.array([[1, 2], [2, 4], [2, 1]])
    #print(convert_2darray_to_1darray(l))
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    page_path = os.path.join(project_dir, 'demo/images/Graduel_de_leglise_de_Nevers-509.nrm.png')
    img = np.array(Image.open(page_path))
    plt.imshow(resize_image(img,4))
    plt.show()


