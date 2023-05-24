import cv2
import numpy as np

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


def apply_opening(img: np.ndarray, kernel_dil=None, kernel_erode=None, show=False):
    """
    Erosion followed by dilation
    """
    if kernel_dil is None:
        kernel_dil = np.ones((3, 3), np.uint8)
    if kernel_erode is None:
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    dilated_img = cv2.dilate(img, kernel_dil, iterations=1)
    opened_img = cv2.erode(dilated_img, kernel_erode, iterations=1)
    return opened_img