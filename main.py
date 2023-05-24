import cv2 
import pytesseract
from PIL import Image

from utils import *

# gray_img = cv2.imread(r'imgs/img.png', cv2.IMREAD_GRAYSCALE)
gray_img = cv2.imread(r'imgs/good_kobel.png', cv2.IMREAD_GRAYSCALE)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
# gray_img = apply_opening(gray_img, kernel_dil=kernel)
gray_img = cv2.erode(gray_img, kernel, iterations=1)
# resized_img = ResizeWithAspectRatio(gray_img, width=400)
resized_img = ResizeWithAspectRatio(gray_img, width=900)
print(pytesseract.image_to_string(gray_img))

cv2.imshow('output', resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()