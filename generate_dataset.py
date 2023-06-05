import os
import csv
import shutil
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from trdg.generators import GeneratorFromRandom

OUTPUT_FOLDER = Path('dataset')
TRAIN_FOLDER = OUTPUT_FOLDER / 'train'
VALID_FOLDER = OUTPUT_FOLDER / 'valid'
LABELS_FILE = 'labels.csv'
FILE_NAME = 'filename'
LABEL = 'words'


def save_image(label, img, str_count):
    file_name = str(str_count) + '.jpg'
    writer.writerow([file_name, label])
    img.save(OUTPUT_FOLDER / file_name)


def generate_images(generator, str_count):
    for img, label in generator:
        save_image(label, img, str_count)
        str_count += 1
    return str_count


def init_directories():
    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(TRAIN_FOLDER)
    os.makedirs(VALID_FOLDER)


def split_dataset():
    df = pd.read_csv(OUTPUT_FOLDER / LABELS_FILE)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    move_images(train_df, TRAIN_FOLDER)
    move_images(val_df, VALID_FOLDER)
    train_df.to_csv(os.path.join(TRAIN_FOLDER, "labels.csv"), index=False)
    val_df.to_csv(os.path.join(VALID_FOLDER, "labels.csv"), index=False)


def move_images(df, directory):
    for index, row in df.iterrows():
        image_name = row[FILE_NAME]
        image_path = os.path.join(OUTPUT_FOLDER, image_name)
        destination_path = os.path.join(directory, image_name)
        shutil.move(image_path, destination_path)


if __name__ == '__main__':
    init_directories()
    # Text on cables mostly contains of letters and numbers
    normal_dataset_size = 80
    generator_without_symbols = GeneratorFromRandom(
        count=normal_dataset_size,
        use_symbols=False,
        fonts=['dot_matrix/DOTMATRI.TTF'],
        size=48
    )

    symbol_dataset_size = 20
    generator_with_symbols = GeneratorFromRandom(
        count=symbol_dataset_size,
        fonts=['dot_matrix/DOTMATRI.TTF'],
        size=48
    )

    with open(OUTPUT_FOLDER / LABELS_FILE, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([FILE_NAME, LABEL])
        str_count = 0
        str_count = generate_images(generator_without_symbols, str_count)
        _ = generate_images(generator_with_symbols, str_count)

    split_dataset()
