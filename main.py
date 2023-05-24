import cv2 
import pytesseract
from PIL import Image

from utils import *

img_cv = cv2.imread(r'imgs/kobel.png')
img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

# resized_img = ResizeWithAspectRatio(img_rgb, width=1180)
# cv2.imshow('output', resized_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
print(pytesseract.image_to_string(img_rgb, config=r'--oem 3 --psm 6'))
#
# img_rgb = Image.frombytes('RGB', img_cv.shape[:2], img_cv, 'raw', 'BGR', 0, 0)
# print(pytesseract.image_to_string(img_rgb))