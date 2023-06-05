import easyocr
import cv2

from utils import *

IMG_FOLDER = 'images'
ground_truth_1 = "PRAFLaSafeX-J5x6REMTPPRAKAB02/99PRAKAB2023|005769m"
ground_truth_2 = "(N)HXCH4x16RE/16FE180/E9OO.6/1kV◁VDE-REG8310▷PRAKAB(€2023|0487m"

def process_img(img_name):
    gray_img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    _, gray_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
    img = remove_black_background(gray_img)
    img = cv2.bitwise_not(img)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    img = apply_iterative_closing(img, kernel_dil=kernel1, kernel_erode=kernel2)
    img = cv2.bitwise_not(img)
    cv2.imwrite(IMG_FOLDER + '/processed_img.png', img)
    return img


def apply_easy_ocr(img, gt):
    reader = easyocr.Reader(['en'])
    results = reader.readtext(img, detail=1, paragraph=False)
    res_string = ''
    for res in results:
        res_string += res[1]
    res_string = res_string.replace(' ', '')
    distance = calculate_levenshtein_distance(gt, res_string)
    max_distance = calculate_levenshtein_distance(gt, '')
    print(f'Recognised: \n {res_string}')
    print(f'Levenshtein distance: {distance}')
    print(distance/max_distance)


if __name__ == '__main__':
    img = process_img(IMG_FOLDER + '/kobel.png')
    apply_easy_ocr(img, ground_truth_1)
