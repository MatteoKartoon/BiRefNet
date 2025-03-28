import argparse
import os
import random
from config import Config

config = Config()

def move_images(image_list, source_dir, dest_dir):
        """
        Move images from source directory to destination directory
        Args:
            image_list: list of image filenames to move
            source_dir: directory to move images from
            dest_dir: directory to move images to
        """
        os.makedirs(dest_dir, exist_ok=True)
        for image in image_list:
            src_path = os.path.join(source_dir, image)
            dest_path = os.path.join(dest_dir, image)
            os.rename(src_path, dest_path)


def split_dataset(train_ratio, val_ratio, test_ratio):
    """
    Split dataset into train, validation and test sets
    Args:
        train_ratio: percentage for training (e.g., 80)
        val_ratio: percentage for validation (e.g., 10)
        test_ratio: percentage for testing (e.g., 10)
    """
    
    # Convert percentages to decimals
    train_split = train_ratio / 100
    val_split = val_ratio / 100
    test_split = test_ratio / 100
    
    # Your dataset splitting logic here
    # For example:
    # 1. Get list of all images
    # 2. Shuffle randomly
    # 3. Split according to ratios
    # 4. Save splits to respective directories

    # Get list of all images
    dataset_path = config.data_root_dir+"/fine_tuning/to_dataset"
    train_dir=dataset_path.replace("to_dataset", "train")
    val_dir=dataset_path.replace("to_dataset", "validation")
    test_dir=dataset_path.replace("to_dataset", "test")
    images = [f for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]

    random.shuffle(images) #to be changed

    train_images = images[:int(len(images) * train_split)]
    val_images = images[int(len(images) * train_split):int(len(images) * (train_split + val_split))]
    test_images = images[int(len(images) * (train_split + val_split)):]

    # Define destination directories
    train_dir = os.path.join(dataset_path, 'train')
    val_dir = os.path.join(dataset_path, 'val')
    test_dir = os.path.join(dataset_path, 'test')

    # Move images to respective directories
    move_images(train_images, dataset_path, train_dir)
    move_images(val_images, dataset_path, val_dir)
    move_images(test_images, dataset_path, test_dir)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split dataset into train/val/test sets')
    parser.add_argument('--train', type=int, default=80, help='Training set percentage')
    parser.add_argument('--val', type=int, default=10, help='Validation set percentage')
    parser.add_argument('--test', type=int, default=10, help='Test set percentage')
    
    args = parser.parse_args()
    
    try:
        split_dataset(args.train, args.val, args.test)
    except ValueError as e:
        print(f"Error: {e}")
