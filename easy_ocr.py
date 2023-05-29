import easyocr
import cv2

from utils import *

ground_truth_1 = "PRAFLaSafeX-J5x6REMTPPRAKAB02/99PRAKAB2023|005769m"
ground_truth_2 = "(N)HXCH4x16RE/16FE180/E9OO.6/1kV◁VDE-REG8310▷PRAKAB(€2023|0487m"
ground_truth_3 = "PRAFLaSafeX-J5x6REMTPPRAKAB02/99PRAKAB2023|005857m"

gray_img = cv2.imread(r'images/kobel_3.png', cv2.IMREAD_GRAYSCALE)
_, gray_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
img = remove_black_background(gray_img)
img = cv2.bitwise_not(img)
kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
img = apply_iterative_closing(img, kernel_dil=kernel1, kernel_erode=kernel2)
img = cv2.bitwise_not(img)
cv2.imwrite('images/im4_it_100.png', img)

reader = easyocr.Reader(['en'])
results = reader.readtext(img, detail=1, paragraph=False)

res_string = ''
for res in results:
    res_string += res[1]

res_string = res_string.replace(' ', '')
print(res_string)
distance = calculate_levenshtein_distance(ground_truth_3, res_string)
print(distance)

