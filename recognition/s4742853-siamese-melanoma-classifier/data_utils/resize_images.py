import os
from PIL import Image
from tqdm import tqdm
import pandas as pd


def resize_dataset(input_dir, output_dir, size=(224, 224)):
    os.makedirs(output_dir, exist_ok=True)

    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith('.jpg'):
            img = Image.open(os.path.join(input_dir, filename))
            img = img.resize(size, Image.Resampling.LANCZOS)
            img.save(os.path.join(output_dir, filename), quality=95)


resize_dataset('../data/cleaned/train_images', '../data/cleaned/train_images_224')
resize_dataset('../data/cleaned/validation_images', '../data/cleaned/validation_images_224')
resize_dataset('../data/cleaned/test_images', '../data/cleaned/test_images_224')