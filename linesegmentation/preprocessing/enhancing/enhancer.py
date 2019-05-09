from scipy.signal import convolve2d
import numpy as np
import cv2


def enhance(image):
    image = cv2.bilateralFilter((image * 255).astype(np.uint8), 5, 75, 75)
    image = (1 - np.clip(convolve2d(1 - image, np.full((1, 10), 0.2), mode="same"), 0, 255)) / 255
    return image


if __name__ == "__main__":
    from PIL import Image
    from matplotlib import pyplot as plt
    import os
    from linesegmentation.preprocessing.binarization.ocropus_binarizer import binarize

    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    page_path = os.path.join(project_dir, 'demo/images/Graduel_de_leglise_de_Nevers-509.nrm.png')
    gray_image = np.array(Image.open(page_path)) / 255.0
    enhanced_image = enhance(gray_image)
    f, ax = plt.subplots(1, 2, True, True)
    ax[0].imshow(gray_image, cmap='gray')
    ax[1].imshow(enhanced_image, cmap='gray')
    plt.show()

    f, ax = plt.subplots(1, 2, True, True)
    ax[0].imshow(binarize(gray_image), cmap='gray')
    ax[1].imshow(binarize(enhanced_image), cmap='gray')
    plt.show()
