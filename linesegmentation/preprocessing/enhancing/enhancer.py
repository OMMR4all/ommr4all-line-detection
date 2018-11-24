from scipy.signal import convolve2d
import numpy as np
import cv2

def enhance(image):
    image = cv2.bilateralFilter((image * 255).astype(np.uint8), 5, 75, 75)
    image = (1 - np.clip(convolve2d(1 - image, np.full((1, 10), 0.2)), 0, 255)) / 255
    return image