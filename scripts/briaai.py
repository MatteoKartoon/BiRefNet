from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import argparse
import os
from transformers import pipeline


def remove_background(input_image_path, output_image_path):
    pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)
    pillow_mask = pipe(input_image_path, return_mask = True) # outputs a pillow mask
    pillow_mask.save(output_image_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Remove background from images in a folder using briaai/RMBG-2.0 model')
    parser.add_argument('--input_folder', '-i', required=True, help='Path to input image folder')
    parser.add_argument('--output_folder', '-o', required=True, help='Path to output image folder')
    
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in os.listdir(input_folder):
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file)
        remove_background(input_path, output_path)