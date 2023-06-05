# Prokap task

There are shorter versions of two main approaches in this repository.

### Applying easy_ocr directly

https://github.com/JaidedAI/EasyOCR
EasyOcr is a tool for ocr :) Some preprocessing of the image is done to improve the recognition results. To use it just run easy_ocr.py module from the repository.
Set ground truth and path to image.

### Train your own model and use it with easy_ocr

1. Generate dataset using dotted matrix font with generate_dataset.py. https://github.com/Belval/TextRecognitionDataGenerator
2. Train your own model by following https://github.com/JaidedAI/EasyOCR/tree/master/trainer. config/train_config.yaml can be used instead of en_filtered_config.yaml.
However, changing number of workers to > 0 is advised to allow parallel processing.
3. Than evaluate the model as described here https://github.com/JaidedAI/EasyOCR/blob/master/custom_model.md
