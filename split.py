import argparse
import os
import random
from config import Config
from typing import List
from datetime import datetime
import shutil
from PIL import Image
import numpy as np

config = Config()


def check_correspondence(gt_images: List[str], original_images: List[str]):
    """
    Check if the images have been split correctly, with each annotated image coinciding with an original image
    Return True if the splitting is correct, False otherwise
    """
    if common_order_1(gt_images, original_images)==gt_images:
        return True
    return False

def move_images(image_list: List[str], source_dir: str, dest_dir: str):
        """
        Move images from source directory to destination directory
        Args:
            image_list: list of images to move
            source_dir: directory to move images from
            dest_dir: directory to move images to
        """
        for image in image_list:
            #for each image find the original path and move it to the destination directory
            if image in os.listdir(source_dir):
                shutil.copy2(os.path.join(source_dir, image), os.path.join(dest_dir, image))
                os.remove(os.path.join(source_dir, image))
            else:
                raise ValueError(f"Could not find source path for image: {image}, while moving images from {source_dir} to {dest_dir}")


def rename_picture(picture_path: str):
    """
    Rename the picture to the format "alphachannel_context_characterid_seed.png"
    Return the new name of the image
    """
    picture_name=picture_path.split("/")[-1]
    picture_name_split=picture_name.split("_")
    new_name="_".join([picture_name_split[0], "-".join(picture_name_split[1:-3]), picture_name_split[-3], ''.join(c for c in picture_name_split[-2] if c.isdigit())])+".png"
    #Check if the new name already exists
    if os.path.exists(os.path.join(os.path.dirname(picture_path), new_name)):
        raise ValueError(f"The image {new_name} has already been added to the list")
    os.rename(picture_path, os.path.join(os.path.dirname(picture_path), new_name))
    return new_name

def special_print(image: str):
    """
    Print the image in a special format
    """
    return "Alpha channel: "+image.split("_")[0]+", Context: "+image.split("_")[1]+", Character ID: "+image.split("_")[2]+", Seed: "+image.split("_")[-1].replace(".png", "").replace("seed", "")

def common_order_1(image_list1: List[str], image_list2: List[str]):
    """
    Return a list containing the common elements of the two lists, order in the same way as the first list
    """
    return [image_list1[i] for i in range(len(image_list1)) if image_list1[i] in image_list2]

def split_dataset(train_ratio: float, val_ratio: float, test_ratio: float, input_dir: str, output_dir: str, gt_dir: str):
    """
    Split dataset into train, validation and test sets
    Args:
        train_ratio: ratio for training (e.g., 0.8)
        val_ratio: ratio for validation (e.g., 0.1)
        test_ratio: ratio for testing (e.g., 0.1)
        input_dir: input directory
        output_dir: output directory
        gt_dir: ground truth directory
    """
    print("Splitting dataset with train_ratio: {}, val_ratio: {}, test_ratio: {}, from the folder {}, to the folder {}, taking as ground truth folder {}".format(train_ratio, val_ratio, test_ratio, input_dir, output_dir, gt_dir))
    
    # Dataset splitting logic

    # 1. Get list of all images
    # 2. Shuffle randomly
    # 3. Split according to ratios
    # 4. Save splits to respective directories
    
    #Get the train, test, validation folders paths
    #If the GT folder contain the word "annotated", then the output folder will contain the word "train", "validation", "test"
    train_dir=os.path.join(output_dir, os.path.basename(gt_dir).replace("annotated", "train"))
    val_dir=os.path.join(output_dir, os.path.basename(gt_dir).replace("annotated", "validation"))
    test_dir=os.path.join(output_dir, os.path.basename(gt_dir).replace("annotated", "test"))

    #Get the paths of the original images and the ground truth
    train_dir_im = os.path.join(train_dir, "im")
    train_dir_gt = os.path.join(train_dir, "an")
    val_dir_im = os.path.join(val_dir, "im")
    val_dir_gt = os.path.join(val_dir, "an")
    test_dir_im = os.path.join(test_dir, "im")
    test_dir_gt = os.path.join(test_dir, "an")

    # Get the list of all ground truth images and original images, renaiming them to the format "alphachannel_context_characterid_seed.png"
    images=[rename_picture(os.path.join(gt_dir, f)) for f in os.listdir(gt_dir) if f.endswith(".png")]
    original_images = [rename_picture(os.path.join(input_dir, f)) for f in os.listdir(input_dir) if f.endswith('.png')]

    #Check one to one correspondency between ground truth and original images
    check=check_correspondence(images, original_images)

    if check:
        print("The images have been split correctly, moving them to the right folders...")
    else:
        raise ValueError(f"Error splitting the images")

    # Shuffle the images to split them randomly
    random.shuffle(images)

    # Split the images
    train_images = images[:int(len(images) * train_ratio)]
    val_images = images[int(len(images) * train_ratio):int(len(images) * (train_ratio + val_ratio))]
    test_images = images[int(len(images) * (train_ratio + val_ratio)):]
    
    #Create the directories to save the train, validation and test sets
    try:
        os.makedirs(train_dir_im, exist_ok=False)
        os.makedirs(val_dir_im, exist_ok=False)
        os.makedirs(test_dir_im, exist_ok=False)
        os.makedirs(train_dir_gt, exist_ok=False)
        os.makedirs(val_dir_gt, exist_ok=False)
        os.makedirs(test_dir_gt, exist_ok=False)
    except FileExistsError:
        print("These split directories already exist, skipping the split...")

    # Save the splitting information
    with open(os.path.join(output_dir, "splitting.log"), "a") as f:
        f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+"\n")
        f.write(f"Splitting the folder {gt_dir.split('/')[-1]} with the following ratios:\n")
        f.write(f"Train ratio: {train_ratio}, Validation ratio: {val_ratio}, Test ratio: {test_ratio}\n")
        f.write(f"Images found in the folder: {len(images)}\n")
        f.write("--------------------------------\n")
        f.write("Images located in the training set:\n")
        for image in train_images:
            f.write(special_print(image)+"\n")
        f.write(f"Total train images: {len(train_images)}\n")
        f.write("--------------------------------\n")
        f.write("Images located in the validation set:\n")
        for image in val_images:
            f.write(special_print(image)+"\n")
        f.write(f"Total validation images: {len(val_images)}\n")
        f.write("--------------------------------\n")
        f.write("Images located in the test set:\n")
        for image in test_images:
            f.write(special_print(image)+"\n")
        f.write(f"Total test images: {len(test_images)}\n")
        f.write("--------------------------------\n")
        f.write("\n")
        f.write("\n")

    # GT
    move_images(train_images, gt_dir, train_dir_gt)
    move_images(val_images, gt_dir, val_dir_gt)
    move_images(test_images, gt_dir, test_dir_gt)

    #Original images
    move_images(train_images, input_dir, train_dir_im)
    move_images(val_images, input_dir, val_dir_im)
    move_images(test_images, input_dir, test_dir_im)

    #Remove the original images and ground truth folders    
    try:
        shutil.rmtree(input_dir)
        shutil.rmtree(gt_dir)
    except FileNotFoundError:
        print(f"Directories already removed or don't exist")

    print("Dataset split successfully")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split dataset into train/val/test sets')
    parser.add_argument('--input_dir', type=str, help='Input directory')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--gt_dir', type=str, help='Ground truth directory')
    
    args = parser.parse_args()
    
    #Splitting with ratio 80/10/10
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    try:
        split_dataset(train_ratio, val_ratio, test_ratio, args.input_dir, args.output_dir, args.gt_dir)
    except ValueError as e:
        print(f"Error: {e}")