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


def check_121_coupling(train_images: List[str], val_images: List[str], test_images: List[str], original_images: List[str]):
    """
    Check if the images have been split correctly, with each annotated image coinciding with anoriginal image
    Return the original images split as well as the ground truth
    """
    #Split the original images as well as the ground truth
    train_images_im = []
    val_images_im = []
    test_images_im = []

    #Try to find for each ground truth an original image, otherwise raise an error
    for image in train_images:
        app_im=[original_image for original_image in original_images if original_image[:4]==image[:4] and original_image[4]=="base"]
        if len(app_im)==1:
            train_images_im.append(app_im[0])
            original_images.remove(app_im[0])
        else:
            raise ValueError(f"Error putting the orginal images into the train set: The image {special_print(image)} has {len(app_im)} corresponding original images, expected 1")
        
    for image in val_images:
        app_im=[original_image for original_image in original_images if original_image[:4]==image[:4] and original_image[4]=="base"]
        if len(app_im)==1:
            val_images_im.append(app_im[0])
            original_images.remove(app_im[0])
        else:
            raise ValueError(f"Error putting the orginal images into the validation set: The image {special_print(image)} has {len(app_im)} corresponding original images, expected 1")
        
    for image in test_images:
        app_im=[original_image for original_image in original_images if original_image[:4]==image[:4] and original_image[4]=="base"]
        if len(app_im)==1:
            test_images_im.append(app_im[0])
            original_images.remove(app_im[0])
        else:
            raise ValueError(f"Error putting the orginal images into the test set: The image {special_print(image)} has {len(app_im)} corresponding original images, expected 1")
        
    #If the splitting is correct, return the split vectors
    print("The images have been split correctly, moving them to the right folders...")
    return train_images_im, val_images_im, test_images_im


def move_images(image_list: List[str], source_dir: str, dest_dir: str):
        """
        Move images from source directory to destination directory
        Args:
            image_list: list of images to move (each image is a list of its details)
            source_dir: directory to move images from
            dest_dir: directory to move images to
        """
        for image in image_list:
            #for each image find the original path and move it to the destination directory
            orig_paths = os.listdir(source_dir)
            src_path = None
            for orig_path in orig_paths:
                new_name = "_".join([image[0], image[1], image[2], image[3]])
                new_name_ass=new_name+"_"+image[4]
                if new_name_ass in orig_path:
                    src_path = os.path.join(source_dir, orig_path)
                    dest_path = os.path.join(dest_dir, new_name+".png")
                    break
            
            if src_path is None:
                raise ValueError(f"Could not find source path for image: {special_print(image)}, while moving images from {source_dir} to {dest_dir}")
                
            shutil.copy2(src_path, dest_path)
            os.remove(src_path)  #if you want to remove original


def save_picture_info(file_name: str):
    """
    Save for the image filename, the information about it
    Args:
        filename: image filename
    """
    # Function to extract relevant parts from the filename
    file_name = file_name.replace(" ", "") #Remove spaces
    parts = file_name.split('_')
    if len(parts) >= 5:
        if len(parts) == 8 and len(parts[5]) == 36 and parts[5].count('-') == 4:
            alpha_value = parts[0]
            context=f"{parts[1]}_{parts[2]}_{parts[3]}_{parts[4]}"
            uuid = parts[5]
            seed = parts[6]
            asset_type = '_'.join(parts[7:]).replace('.png', '')
            return alpha_value, context, uuid, seed, asset_type
        elif len(parts[3]) == 36 and parts[3].count('-') == 4:
            alpha_value = parts[0]
            context = f"{parts[1]}_{parts[2]}"
            uuid = parts[3]
            seed = parts[4]
            asset_type = '_'.join(parts[5:]).replace('.png', '')
            return alpha_value, context, uuid, seed, asset_type
        elif len(parts[4]) == 36 and parts[4].count('-') == 4:
            alpha_value = parts[0]
            context = f"{parts[1]}_{parts[2]}_{parts[3]}"
            uuid = parts[4]
            seed = parts[5]
            asset_type = '_'.join(parts[6:]).replace('.png', '')
            return alpha_value, context, uuid, seed, asset_type
        elif len(parts[2]) == 36 and parts[2].count('-') == 4:
            alpha_value = parts[0]
            context = parts[1]
            uuid = parts[2]
            seed = parts[3]
            asset_type = '_'.join(parts[4:]).replace('.png', '')
            return alpha_value, context, uuid, seed, asset_type
        else:
            raise ValueError(f"Filename {file_name} does not match expected format (alpha_value_context_uuid_seed_asset_type.png), got parts: {parts}")
    return None



def special_print(image: List[str]):
    """
    Print the image information in a special format
    Args:
        image: image details vector
    """
    return f"Alpha value: {image[0]}, Context: {image[1]}, UUID: {image[2]}, Seed: {image[3]}"



def split_dataset(train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1):
    """
    Split dataset into train, validation and test sets
    Args:
        train_ratio: ratio for training (e.g., 0.8)
        val_ratio: ratio for validation (e.g., 0.1)
        test_ratio: ratio for testing (e.g., 0.1)
    """
    print("Splitting dataset with train_ratio: {}, val_ratio: {}, test_ratio: {}".format(train_ratio, val_ratio, test_ratio))
    
    # Dataset splitting logic

    # 1. Get list of all images
    # 2. Shuffle randomly
    # 3. Split according to ratios
    # 4. Save splits to respective directories

    # Get the folders paths
    data_folder = config.data_root_dir+"/fine_tuning/"
    
    # Get the list of all folders in the data folder
    for folder in os.listdir(data_folder):
        #If the folder is a folder of annotated images
        if folder.startswith("annotated"):
            dataset_path_gt = os.path.join(data_folder, folder)
            
            #Get the train, test, validation folders paths
            train_dir=os.path.join(data_folder, os.path.basename(dataset_path_gt).replace("annotated", "train"))
            val_dir=os.path.join(data_folder, os.path.basename(dataset_path_gt).replace("annotated", "validation"))
            test_dir=os.path.join(data_folder, os.path.basename(dataset_path_gt).replace("annotated", "test"))

            #Specify the paths of the original images folder
            dataset_path_im = dataset_path_gt.replace("annotated", "filtred")

            #Get the paths of the images and the ground truth
            train_dir_im = os.path.join(train_dir, "im")
            train_dir_gt = os.path.join(train_dir, "gt")
            val_dir_im = os.path.join(val_dir, "im")
            val_dir_gt = os.path.join(val_dir, "gt")
            test_dir_im = os.path.join(test_dir, "im")
            test_dir_gt = os.path.join(test_dir, "gt")

            # Get the list of all ground truth images and original images, saving their details
            images=[]
            for f in os.listdir(dataset_path_gt):
                if f.endswith(".png"):
                    if save_picture_info(f) in images:
                        raise ValueError(f"The image {special_print(f)} has already been added to the list")
                    images.append(save_picture_info(f))

            original_images = [save_picture_info(f) for f in os.listdir(dataset_path_im) if f.endswith('.png')]

            # Shuffle the images to split them randomly
            random.shuffle(images)

            # Split the images
            train_images = images[:int(len(images) * train_ratio)]
            val_images = images[int(len(images) * train_ratio):int(len(images) * (train_ratio + val_ratio))]
            test_images = images[int(len(images) * (train_ratio + val_ratio)):]

            train_images_im, val_images_im, test_images_im = check_121_coupling(train_images, val_images, test_images, original_images)
            
            #Create the directories to save the train, validation and test sets
            try:
                os.makedirs(train_dir_im, exist_ok=False)
                os.makedirs(val_dir_im, exist_ok=False)
                os.makedirs(test_dir_im, exist_ok=False)
                os.makedirs(train_dir_gt, exist_ok=False)
                os.makedirs(val_dir_gt, exist_ok=False)
                os.makedirs(test_dir_gt, exist_ok=False)
            except FileExistsError:
                print("These directories have already been split")
                break

            # Save the splitting information
            with open(os.path.join(data_folder, "splitting.log"), "a") as f:
                f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+"\n")
                f.write(f"Splitting the folder {dataset_path_gt.split('/')[-1]} with the following ratios:\n")
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

            #Original images
            move_images(train_images_im, dataset_path_im, train_dir_im)
            move_images(val_images_im, dataset_path_im, val_dir_im)
            move_images(test_images_im, dataset_path_im, test_dir_im)

            # GT
            move_images(train_images, dataset_path_gt, train_dir_gt)
            move_images(val_images, dataset_path_gt, val_dir_gt)
            move_images(test_images, dataset_path_gt, test_dir_gt)

            #Remove the original images and ground truth folders    
            try:
                shutil.rmtree(dataset_path_im)
                shutil.rmtree(dataset_path_gt)
            except FileNotFoundError:
                print(f"Directories already removed or don't exist")

            print("Dataset split successfully")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split dataset into train/val/test sets')
    parser.add_argument('--train', type=float, default=0.8, help='Training set percentage')
    parser.add_argument('--val', type=float, default=0.1, help='Validation set percentage')
    parser.add_argument('--test', type=float, default=0.1, help='Test set percentage')
    
    args = parser.parse_args()
    
    try:
        split_dataset(args.train, args.val, args.test)
    except ValueError as e:
        print(f"Error: {e}")