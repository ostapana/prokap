import cv2
import numpy as np
import Levenshtein

BLACK = 0
WHITE = 255


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def apply_closing(img: np.ndarray, kernel_dil=None, kernel_erode=None, show=False):
    """
    Dilation followed by erosion
    """
    if kernel_erode is None:
        kernel_erode = np.ones((3, 3), np.uint8)
    if kernel_dil is None:
        kernel_dil = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    dilated_img = cv2.dilate(img, kernel_dil, iterations=2)
    closed_img = cv2.erode(dilated_img, kernel_erode, iterations=1)
    dilated_img = cv2.dilate(closed_img, kernel_dil, iterations=1)
    closed_img = cv2.erode(dilated_img, kernel_erode, iterations=2)
    dilated_img = cv2.dilate(closed_img, kernel_dil, iterations=2)
    closed_img = cv2.erode(dilated_img, kernel_erode, iterations=2)
    return closed_img


def apply_iterative_closing(img: np.ndarray, kernel_dil=None, kernel_erode=None):
    for i in range(10):
        img = cv2.dilate(img, kernel_dil, iterations=1)
        img = cv2.erode(img, kernel_erode, iterations=1)
    return img


def remove_black_background(img: np.ndarray) -> np.ndarray:
    new_img = np.ndarray(shape=(0, img.shape[1]), dtype=np.uint8)
    for row in img:
        if not (row == 0).all():
            new_img = np.concatenate((new_img, row[np.newaxis, ...]))
    return new_img


def calculate_levenshtein_distance(str1, str2):
    distance = Levenshtein.distance(str1, str2)
    return distance
