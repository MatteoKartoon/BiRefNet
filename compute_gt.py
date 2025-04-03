import os
import argparse
from PIL import Image
import numpy as np

def compute_gt(image_path):
    # Open the image
    img = Image.open(image_path)
    
    # Convert to RGBA if not already
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # Get the alpha channel
    alpha_channel = img.split()[3]  # Get the alpha channel (4th channel)
    
    # Convert alpha channel to grayscale
    gray_image = alpha_channel.convert('L')
    
    return gray_image

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            image = Image.open(input_path)
            alpha_mask = compute_gt(input_path)

            # Save the alpha mask
            alpha_mask.save(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute alpha channel masks for images.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input folder containing images.')
    parser.add_argument('--output', type=str, required=True, help='Path to the output folder to save alpha masks.')
    args = parser.parse_args()

    process_images(args.input, args.output)