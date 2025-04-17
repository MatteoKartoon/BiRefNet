import argparse
from PIL import Image
import os
import argparse
from glob import glob
import prettytable as pt
import matplotlib
import random
import copy
import cv2

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from evaluation.metrics import evaluator
from config import Config

config = Config()

#Function to compute the green background image, given the original image and a mask
def pt_green_bg(original, mask):
    # Open the original image and the mask
    original_img = Image.open(original).convert("RGBA")
    mask_img = Image.open(mask).convert("L")

    # Create a new image with the same size as the original, filled with the green background color
    green_bg = Image.new("RGBA", original_img.size, (0, 255, 17, 255))

    # Composite the original image onto the green background using the mask
    result_img = Image.composite(original_img, green_bg, mask_img)

    return result_img


def pt_red_pixels(gt: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Creates a visualization image highlighting differences between ground truth and mask.
    
    Args:
        gt (np.ndarray): Ground truth image as a numpy array (grayscale)
        mask (np.ndarray): Predicted mask as a numpy array (grayscale)
        
    Returns:
        np.ndarray: represent an RGB image where:
            - Red channel shows large differences (>10) between gt and mask
            - Grayscale shows small differences (≤10) using mask values
            
    Note:
        The function expects grayscale images (2D arrays) and returns an RGB image (3D array)
    """
    # Ensure input images have the same dimensions
    assert gt.size == mask.size

    # Create output array with shape (height, width, 3) for RGB
    output = np.zeros((*gt.shape, 3), dtype=np.uint8)
    
    # Calculate absolute pixel-wise differences between ground truth and mask
    diff = np.abs(gt.astype(np.int16) - mask.astype(np.int16))
    
    # Create boolean mask for pixels where difference exceeds threshold
    large_diff_mask = diff > 10
    
    # For large differences: set red channel to difference value
    # This creates red highlights where predictions differ significantly
    output[large_diff_mask, 0] = 255
    
    # For small differences: set all RGB channels to mask value
    # This creates grayscale areas where predictions match well
    output[~large_diff_mask, 0] = mask[~large_diff_mask]  # R
    output[~large_diff_mask, 1] = mask[~large_diff_mask]  # G
    output[~large_diff_mask, 2] = mask[~large_diff_mask]  # B

    #check why there are some pixels that are not red or gray scale
    #SOLVED:Even if your pixel values are precisely red [255, 0, 0] or grayscale [x, x, x], Image.fromarray might internally perform slight interpolation or color space interpretation when you resize, save, or display the image — especially on screens that auto-apply gamma correction.
    #c=0
    #for i in range(output.shape[0]):
    #    for j in range(output.shape[1]):
    #        if not ((output[i, j, 0] == 255) & (output[i, j, 1] == 0) & (output[i, j, 2] == 0)) and not ((output[i, j, 0] == output[i, j, 1]) & (output[i, j, 0] == output[i, j, 2])):
    #            c+=1
    #print(f"Number of pixels that are not red or gray scale: {c}")
    #print(output.shape)
    
    return output


def erode_red(image: np.ndarray) -> np.ndarray:
    """
    Erode the red channel of the image, and color in green the eroded pixels
    """
    kernel = np.ones((5, 5), np.uint8)

    # Create a matrix where for each pixel get 1 if it is a red pixel in image and 0 if it is a grayscale pixel
    red_pixel_matrix = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    
    # Red pixels have a high value in the red channel and low values in the green and blue channels
    red_pixel_mask = (image[:, :, 0] == 255) & (image[:, :, 1] == 0) & (image[:, :, 2] == 0)
    
    # Set a matrix with only red pixels
    red_pixel_matrix[red_pixel_mask] = 1

    # Erode the red pixels
    eroded_image = cv2.erode(red_pixel_matrix, kernel, iterations=1)

    # Get the green pixels that have been eroded
    green_pixel_mask = red_pixel_matrix != eroded_image

    #Final image with not eroded red pixels, and eroded green pixels
    final_image=image.copy()
    final_image[green_pixel_mask, 0] = 0
    final_image[green_pixel_mask, 1] = 255

    return final_image


def compute_interest(metric: str, pred_scores: dict[str], many: int, images_number: int):
    """For a given metric, compute the most interesting images based on the difference between the max and min score got, comparing the different models
    Return the indexes of the most many interesting images"""

    # Initialize a dictionary to store the interest values for each image
    interest_values = [0] * images_number

    # Loop through each image
    for i in range(images_number):
        # Get the scores for the current image from all models
        scores = [pred_scores[model][i][metric] for model in pred_scores]

        # Calculate the difference between the max and min score
        max_score = max(scores)
        min_score = min(scores)
        interest_values[i] = max_score - min_score

    interest_indexes = sorted(range(len(interest_values)), key=lambda i: interest_values[i], reverse=True)[:many]
    #Return the indexes of the most interesting images
    return interest_indexes


def do_visualization(model_paths: list[str], gt_paths: list[str], image_paths: list[str]):
    """Take a list of model paths, ground truth paths and original image paths and,
    print a picture for each model, containing a comparison between the ground truth and the model prediction"""
    
    #Loop through all the models
    for model_path in model_paths:
        print("Visualizing model results: ", model_path)

        # Load the model predictions
        pred_data_dir=os.path.join("e_preds", model_path, args.testset)
        assert os.path.exists(pred_data_dir), "Model prediction path does not exist"
        pred_content = sorted([os.path.join(pred_data_dir, f) for f in os.listdir(pred_data_dir)])
        assert len(pred_content) == len(gt_paths), "Number of model predictions and ground truth paths do not match"
        
        #get the number of model predictions and number of images to be visualized 
        gt_len = len(gt_paths)
        picture_len = min(10,gt_len)

        #initialize the figure
        plt.figure(figsize=(25, 8*picture_len))
        
        #Choose at most 10 random images to be visualized
        visualize_inds=random.sample(range(0,gt_len),picture_len)
        
        #select the pictures to be visualized
        visualize_pictures = [(pred_content[i], gt_paths[i], image_paths[i]) for i in visualize_inds]

        # Evaluate model predictions against ground truth
        for i, (pred_path, gt_path, image_path) in enumerate(visualize_pictures):
            #Evaluate the model predictions against the ground truth
            em, sm, fm, mae, mse, wfm, hce, mba, biou, pa = evaluator(
                gt_paths=[gt_path],
                pred_paths=[pred_path],
                #metrics=args.metrics.split('+'), if we want display only few metrics
                verbose=config.verbose_eval
            )
            #Save the scores for the current image
            scores = [
                    fm['curve'].max().round(3), wfm.round(3), mae.round(3), sm.round(3), em['curve'].mean().round(3), int(hce.round()), 
                    em['curve'].max().round(3), fm['curve'].mean().round(3), em['adp'].round(3), fm['adp'].round(3),
                    mba.round(3), biou['curve'].max().round(3),mse.round(3), biou['curve'].mean().round(3), pa.round(3)
            ]
            #Display
            plt.subplot(picture_len, 6, 6*i+1)
            plt.imshow(Image.open(image_path))
            plt.axis('off')
            plt.title('Original')

            plt.subplot(picture_len, 6, 6*i+3)
            plt.imshow(pt_green_bg(image_path, pred_path))
            plt.axis('off')
            plt.title('Model prediction')

            plt.subplot(picture_len, 6, 6*i+2)
            plt.imshow(pt_green_bg(image_path, gt_path))
            plt.axis('off')
            plt.title('Ground truth')
            
            plt.subplot(picture_len, 6, 6*i+5)
            #plt.imshow(pt_red_pixels(cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE), cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)))
            plt.imshow(Image.open(pred_path))
            plt.axis('off')
            plt.title('Model prediction mask')

            plt.subplot(picture_len, 6, 6*i+4)
            plt.imshow(Image.open(gt_path))
            plt.axis('off')
            plt.title('Ground truth mask')

            plt.subplot(picture_len, 6, 6*i+6)
            plt.axis('off')
            plt.text(0.5, 0.5, f"maxFm: {scores[0]}\nwFmeasure: {scores[1]}\nMAE: {scores[2]}\nSmeasure: {scores[3]}\n"
                    f"meanEm: {scores[4]}\nHCE: {scores[5]}\nmaxEm: {scores[6]}\nmeanFm: {scores[7]}\n"
                    f"adpEm: {scores[8]}\nadpFm: {scores[9]}\nmBA: {scores[10]}\nmaxBIoU: {scores[11]}\nMSE: {scores[12]}\nmeanBIoU: {scores[13]}\npixAcc: {scores[14]}",
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=16)

        plt.tight_layout()
        # Save the figure
        output_file = os.path.join("e_results", f'visualization__{model_path}__{args.testset}.png')
        print(f"Saving visualization to: {output_file}")
        plt.savefig(output_file, bbox_inches='tight', dpi=300)        
        plt.close()
    

def do_ranking(model_paths: list[str], metrics: list[str], gt_paths: list[str], image_paths: list[str]):
    """Take a list of model paths, metrics, ground truth paths and original image paths and,
    print a picture for each specified metric, containing a comparison between the specified models"""
    
    print("Ranking models", model_paths, "based on metrics: ", metrics, "...")

    #define a dictionary having as keys the model paths and as values the paths of the model predictions
    pred_content = {}
    for model_path in model_paths:
        # Load the model predictions
        pred_data_dir=os.path.join("e_preds", model_path, args.testset)
        assert os.path.exists(pred_data_dir), "Model prediction path does not exist"
        model_file_list = sorted([os.path.join(pred_data_dir, f) for f in os.listdir(pred_data_dir)])
        assert len(model_file_list) == len(gt_paths), "Number of model predictions and ground truth paths do not match"
        pred_content[model_path] = model_file_list
    
    #get the number of model predictions and number of images to be visualized 
    gt_len = len(gt_paths)
    picture_len = min(10,gt_len)
    picture_wids=len(model_paths)+2
    
    #Create a dictionary with the same shape as prediction_content dictionary, to save scores for each model and image
    pred_scores = copy.deepcopy(pred_content)
    #Loop through all the images and models, and compute the scores for each model and metric
    for im_ind, im in enumerate(gt_paths):
        for model_path in model_paths:
            pred_path = pred_content[model_path][im_ind]
            #Evaluate the model predictions against the ground truth
            em, sm, fm, mae, mse, wfm, hce, mba, biou, pa = evaluator(
                gt_paths=[im],
                pred_paths=[pred_path],
                metrics=metrics,
                verbose=config.verbose_eval
            )
            #Save the scores for the current image and model in a dictionary
            scores = {'S': sm.round(3), 'MAE': mae.round(3), 'E': em['curve'].mean().round(3), 'F': fm['curve'].mean().round(3), 'WF': wfm.round(3),
                      'MBA': mba.round(3), 'BIoU': biou['curve'].mean().round(3), 'MSE': mse.round(3), 'HCE': int(hce.round()), 'PA': pa.round(3)}
            pred_scores[model_path][im_ind] = scores

    #loop through all the metrics, and do a different visualization for each of them
    for metric in metrics:
        #Compute the most interesting picture_len images based on the metrics, and keep only them
        visualize_inds=compute_interest(metric,pred_scores,picture_len,gt_len)
        #select the pictures to be visualized
        visualize_pictures = [(np.transpose([pred_content[model_path][i] for model_path in model_paths]), gt_paths[i], image_paths[i]) for i in visualize_inds]

        #initialize the figure
        plt.figure(figsize=(5*picture_wids, 6*picture_len))
        #Loop through all the interesting images, the models and the metrics
        for im_ind, (p, g, m) in enumerate(visualize_pictures):
            image_original=Image.open(m)
            image_gt=Image.open(g)
            
            # Display
            plt.subplot(picture_len, picture_wids, picture_wids*im_ind+1)
            plt.imshow(image_original)
            plt.axis('off')
            plt.title('Original',fontsize=20)

            plt.subplot(picture_len, picture_wids, picture_wids*im_ind+2)
            plt.imshow(image_gt)
            plt.axis('off')
            plt.title('Ground truth',fontsize=20)
            
            #for each model show the prediction
            for i in range(len(p)):
                plt.subplot(picture_len, picture_wids, picture_wids*im_ind+3+i)

                # Read as cv2
                gt_img = cv2.imread(g, cv2.IMREAD_GRAYSCALE)
                pred_img = cv2.imread(p[i], cv2.IMREAD_GRAYSCALE)

                #if metric is PA show the red pixels eroded
                if metric!='PA':
                    plt.imshow(Image.fromarray(pt_red_pixels(gt_img, pred_img)))
                else:
                    plt.imshow(Image.fromarray(erode_red(pt_red_pixels(gt_img, pred_img))))

                plt.axis('off')
                tit=p[i].split("/")[1]
                #Pick the correct score from the scores tensor
                score=pred_scores[model_paths[i]][visualize_inds[im_ind]][metric]
                #Display the score
                plt.title("Model "+tit+"\n"+f"{metric}: {score}",fontsize=20)

        plt.tight_layout()

        #save the figure
        output_file = os.path.join("e_results", f'comparison_{metric}__{args.testset}.png')
        print(f"Saving comparison for {metric} to: {output_file}")
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()

if __name__ == '__main__':
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Run visualization or ranking based on the provided parameters.')
    parser.add_argument('--models', type=str, required=True, help='Path to the models')
    parser.add_argument('--metrics', type=str, help='Metrics to be used',
                        default='+'.join(['S', 'MAE', 'E', 'F', 'WF', 'MBA', 'BIoU', 'MSE', 'HCE','PA']))
    parser.add_argument('--testset', type=str, help='Testset to be used', required=True)

    # Parse the arguments
    args = parser.parse_args()

    models = args.models.split('+')
    metrics = args.metrics.split('+')
    testset = os.path.join(config.data_root_dir, "fine_tuning", args.testset)

    assert os.path.exists(testset+"/gt"), f"Ground-truth path {testset}/gt does not exist"
    assert os.path.exists(testset+"/im"), f"Original image path {testset}/im does not exist"

    gt_paths = sorted(glob(os.path.join(testset, 'gt', '*')))
    image_paths = sorted(glob(os.path.join(testset, 'im', '*')))

    assert len(gt_paths) == len(image_paths), f"Number of ground-truth and original image paths {len(gt_paths)} and {len(image_paths)} do not match"

    # Call the appropriate function based on the comparison flag
    do_ranking(models, metrics, gt_paths, image_paths)
    do_visualization(models, gt_paths, image_paths)