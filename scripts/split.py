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
    #Check if there are duplicate elements in one of the two lists
    if len(gt_images) != len(set(gt_images)) or len(original_images) != len(set(original_images)):
        print("Two images with the same name found:")
        duplicate_elements = [item for item in set(gt_images+original_images) if gt_images.count(item) == 2 or original_images.count(item) == 2]
        print(duplicate_elements)
        return False
    #Check if the common elements are in the same position in both lists
    common_images = get_common_elements_of_lists(gt_images, original_images)
    if common_images==gt_images:
        return True
    else:
        print("The following images are not in the original images:")
        for image in gt_images:
            if image not in common_images:
                print(f"\"{image}\"")
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


def new_name(picture_name: str):
    """
    Takes as input a picture name
    Return the new name of the image in the format "alphachannel_context_characterid_seed.png"
    """
    picture_name_split=picture_name.split("_")
    #Check if there is the word "seed" in the name
    if picture_name_split[-2].startswith("seed"):
        new_name="_".join([picture_name_split[0], "-".join(picture_name_split[1:-3]), picture_name_split[-3], picture_name_split[-2] ])+".png"
    else:
        new_name="_".join([picture_name_split[0], "-".join(picture_name_split[1:-2]), picture_name_split[-2]])+".png"
    return new_name

def rename_picture_list(old_names: List[str], new_names: List[str], folder_path: str):
    """
    Takes as input a list of old names, a list of new names and the path of the folder where images are stored
    Find the old name pictures and rename them to the new name
    """
    for old_name, new_name in zip(old_names, new_names):
        os.rename(os.path.join(folder_path,old_name), os.path.join(folder_path, new_name))
    

def special_print(image: str):
    """
    Print the image in a special format
    """
    #Check if there is the word "seed" in the name
    if image.split("_")[-1].startswith("seed"):
        return "Alpha channel: "+image.split("_")[0]+", Context: "+"_".join(image.split("_")[1:-2])+", Character ID: "+image.split("_")[-2]+", Seed: "+image.split("_")[-1].replace(".png", "").replace("seed", "")
    return "Alpha channel: "+image.split("_")[0]+", Context: "+"_".join(image.split("_")[1:-1])+", Character ID: "+image.split("_")[-1]

def get_common_elements_of_lists(image_list1: List[str], image_list2: List[str]):
    """
    Return a list containing the common elements of the two lists, order in the same way as the first list
    """
    return [image_list1[i] for i in range(len(image_list1)) if image_list1[i] in image_list2]

def split_dataset(train_ratio: float, val_ratio: float, test_ratio: float, input_dir: str, output_dir: str, gt_dir: str, dataset_name: str, seed: int=0):
    """
    Split dataset into train, validation and test sets
    Args:
        train_ratio: ratio for training (e.g., 0.8)
        val_ratio: ratio for validation (e.g., 0.1)
        test_ratio: ratio for testing (e.g., 0.1)
        input_dir: input directory
        output_dir: output directory
        gt_dir: ground truth directory
        dataset_name: name of the dataset
        seed: seed for the random list shuffling
    """
    print("Splitting dataset {} with train_ratio: {}, val_ratio: {}, test_ratio: {}, from the folder {}, to the folder {}, taking as ground truth folder {}".format(dataset_name,train_ratio, val_ratio, test_ratio, input_dir, output_dir, gt_dir))
    
    # Dataset splitting logic

    # 1. Get list of all images
    # 2. Shuffle randomly
    # 3. Split according to ratios
    # 4. Save splits to respective directories
    
    #Get the train, test, validation folders paths
    #Call the directories path where the dataset is split as task_dataset_name_ where task is "train", "validation" or "test"
    splitting_dirs=[os.path.join(output_dir, "train_"+dataset_name),os.path.join(output_dir, "validation_"+dataset_name),os.path.join(output_dir, "test_"+dataset_name)]
    #Get the an and im subdirectories paths
    splitting_dirs_gt=[os.path.join(s,"an") for s in splitting_dirs]
    splitting_dirs_im=[os.path.join(s,"im") for s in splitting_dirs]

    # Get the list of all ground truth images and original images, renaiming them to the format "alphachannel_context_characterid_seed.png"
    found_gt=os.listdir(gt_dir)
    found_im=os.listdir(input_dir)
    #Remove the files that are not png
    found_gt=[f for f in found_gt if f.endswith(".png")]
    found_im=[f for f in found_im if f.endswith(".png")]
    #Do the list of the images in the folder, called with a new name in a specific format
    images=[new_name(f) for f in found_gt]
    original_images = [new_name(f) for f in found_im]

    #Check one to one correspondency between ground truth and original images
    check=check_correspondence(images, original_images)

    #If an error is found, the script will stop, without doing the splitting
    if check:
        print("The images have been split correctly, moving them to the right folders...")
    else:
        raise ValueError(f"Error splitting the images")
    
    #Rename the images to the new format
    rename_picture_list(found_gt,images,gt_dir)
    rename_picture_list(found_im,original_images,input_dir)

    # Shuffle the images to split them randomly
    random.seed(seed)
    random.shuffle(images)

    # Split the images
    train_images = images[:int(len(images) * train_ratio)]
    val_images = images[int(len(images) * train_ratio):int(len(images) * (train_ratio + val_ratio))]
    test_images = images[int(len(images) * (train_ratio + val_ratio)):]
    
    #Create the directories to save the train, validation and test sets
    try:
        for dir_im, dir_gt in zip(splitting_dirs_im, splitting_dirs_gt):
            os.makedirs(dir_im, exist_ok=False)
            os.makedirs(dir_gt, exist_ok=False)
    except FileExistsError:
        raise ValueError("These split directories already exist, skipping the split...")

    # Save the splitting information
    with open(os.path.join(output_dir, "splitting.log"), "a") as f:
        f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+"\n")
        f.write(f"Splitting the dataset {dataset_name} with the following ratios:\n")
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

    # Move the images to the right folders
    split_images=[train_images, val_images, test_images]
    for split_image, dir_im, dir_gt in zip(split_images, splitting_dirs_im, splitting_dirs_gt):
        move_images(split_image, gt_dir, dir_gt)
        move_images(split_image, input_dir, dir_im)

    #Remove the original images and ground truth folders    
    try:
        shutil.rmtree(input_dir)
        shutil.rmtree(gt_dir)
    except FileNotFoundError:
        print(f"Directories already removed or don't exist")

    print("Dataset split successfully")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split dataset into train/val/test sets')
    parser.add_argument('--dataset_name', type=str, help='Dataset name')
    parser.add_argument('--input_dir', type=str, help='Input directory')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--gt_dir', type=str, help='Ground truth directory')
    
    args = parser.parse_args()
    
    #Splitting with ratio 80/10/10
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    try:
        split_dataset(train_ratio, val_ratio, test_ratio, args.input_dir, args.output_dir, args.gt_dir, args.dataset_name)
    except ValueError as e:
        print(f"Error: {e}")