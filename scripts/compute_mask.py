import os
import argparse
from PIL import Image
import numpy as np

def compute_mask(image_path: str, low_threshold: int = 0, high_threshold: int = 220) -> Image.Image:
    """
    Compute the alpha channel mask for an image, setting all pixels below low_threshold to 0 and above high_threshold to 255.
    """
    # Open the image
    img = Image.open(image_path)
    
    # Convert to RGBA if not already
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # Get the alpha channel
    alpha_channel = img.split()[3]  # Get the alpha channel (4th channel)

    # Convert alpha channel to numpy array
    alpha_channel = np.array(alpha_channel)
    
    #Set all pixels below low_threshold to 0 and above high_threshold to 255
    alpha_channel[alpha_channel <= low_threshold] = 0
    alpha_channel[alpha_channel >= high_threshold] = 255

    # Convert alpha channel to grayscale
    gray_image = Image.fromarray(alpha_channel)
    
    return gray_image

def process_images(input_folder, output_folder, low_threshold, high_threshold):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            mask = compute_mask(input_path, low_threshold, high_threshold)

            # Save the alpha mask
            mask.save(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute alpha channel masks for images.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input folder containing images.')
    parser.add_argument('--output', type=str, required=True, help='Path to the output folder to save alpha masks.')
    parser.add_argument('--low_threshold', type=int, default=0, help='Low threshold for the alpha channel.')
    parser.add_argument('--high_threshold', type=int, default=220, help='High threshold for the alpha channel.')
    args = parser.parse_args()

    process_images(args.input, args.output, args.low_threshold, args.high_threshold)