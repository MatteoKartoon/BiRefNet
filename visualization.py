import argparse
from PIL import Image
import os
import argparse
from glob import glob
import prettytable as pt
import matplotlib
import random
import copy

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

def compute_interest(scoresss, metrics,many):
    #Compute an interesting rate for each image based on the difference between the max and min score of the metrics
    print(f"Computing the most interesting {many} images based on metrics: {metrics}...")
    interest_values=[]
    s_copy=copy.deepcopy(scoresss)
    #Loop through all the images
    for scoress in s_copy:
        for ins,scores in enumerate(scoress):
            scoress[ins] = [scores[i] for i in metrics]
        print(scoress)
        int_val=max(map(max, scoress))-min(map(min, scoress)) #Difference between the max and min score for that image
        interest_values.append(int_val)
        print(int_val)
    print(interest_values)
    interest_indexes = sorted(range(len(interest_values)), key=lambda i: interest_values[i], reverse=True)[:many]
    #Return the indexes of the most interesting images
    return interest_indexes

def do_visualization(model_paths):
    #function to visualize the results of the models with the scores
    dm=['S', 'MAE', 'E', 'F', 'WF', 'MBA', 'BIoU', 'MSE', 'HCE']
    #Create a list of the ground truth and image paths
    #gt_paths = sorted(glob(os.path.join(config.testsets, 'gt', '*')))
    #image_paths = sorted(glob(os.path.join(config.testsets, 'im', '*')))
    to_delete=config.testsets.replace("test","train")
    gt_paths = sorted(glob(os.path.join(to_delete, 'gt', '*')))
    image_paths = sorted(glob(os.path.join(to_delete, 'im', '*')))
    #Loop through all the models
    for model_path in model_paths:
        print("Visualizing model results: ", model_path)
        # Load the model predictions
        pred_data_dir = [os.path.join("e_preds", model_path, file_name) for file_name in os.listdir("e_preds/{}".format(model_path))]
        
        pred_data_dir = sorted(pred_data_dir)
        print(pred_data_dir)
        lun = len(pred_data_dir)
        plt.figure(figsize=(25, 8*lun))
        
        #Choose at most 10 random images to visualize
        rs=random.sample(range(0,len(gt_paths)),min(10,len(gt_paths)))
        print(rs)
        zs=list(zip(np.transpose(pred_data_dir), gt_paths, image_paths))
        z=[]
        for r in rs:
            z.append(zs[r])
        # Evaluate model predictions against ground truth
        for i, (pred_path, gt_path, image_path) in enumerate(zip(pred_data_dir, gt_paths, image_paths)):
            print('\t', 'Evaluating prediction: {} against ground truth: {}'.format(pred_path, gt_path))
            #Evaluate the model predictions against the ground truth
            em, sm, fm, mae, mse, wfm, hce, mba, biou = evaluator(
                gt_paths=[gt_path],
                pred_paths=[pred_path],
                metrics=dm,
                verbose=config.verbose_eval
            )
            #Save the scores for the current image
            scores = [
                    fm['curve'].max().round(3), wfm.round(3), mae.round(3), sm.round(3), em['curve'].mean().round(3), int(hce.round()), 
                    em['curve'].max().round(3), fm['curve'].mean().round(3), em['adp'].round(3), fm['adp'].round(3),
                    mba.round(3), biou['curve'].max().round(3),mse.round(3), biou['curve'].mean().round(3),
            ]
            print(scores)
            #Display
            plt.subplot(lun, 6, 6*i+1)
            plt.imshow(Image.open(image_path))
            plt.axis('off')
            plt.title('Original')

            plt.subplot(lun, 6, 6*i+2)
            plt.imshow(pt_green_bg(image_path, pred_path))
            plt.axis('off')
            plt.title('Model prediction')

            plt.subplot(lun, 6, 6*i+3)
            plt.imshow(pt_green_bg(image_path, gt_path))
            plt.axis('off')
            plt.title('Ground truth')
            
            plt.subplot(lun, 6, 6*i+4)
            plt.imshow(Image.open(pred_path))
            plt.axis('off')
            plt.title('Model prediction mask')

            plt.subplot(lun, 6, 6*i+5)
            plt.imshow(Image.open(gt_path))
            plt.axis('off')
            plt.title('Ground truth mask')

            plt.subplot(lun, 6, 6*i+6)
            plt.axis('off')
            plt.text(0.5, 0.5, f"maxFm: {scores[0]}\nwFmeasure: {scores[1]}\nMAE: {scores[2]}\nSmeasure: {scores[3]}\n"
                    f"meanEm: {scores[4]}\nHCE: {scores[5]}\nmaxEm: {scores[6]}\nmeanFm: {scores[7]}\n"
                    f"adpEm: {scores[8]}\nadpFm: {scores[9]}\nmBA: {scores[10]}\nmaxBIoU: {scores[11]}\nMSE: {scores[12]}\nmeanBIoU: {scores[13]}",
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=16)

        plt.tight_layout()
        # Save the figure
        output_file = os.path.join("e_results", f'visualization_{model_path.replace("/", "_")}.png')
        print(f"Saving visualization to: {output_file}")
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        
        # Try to display (will work in interactive environments)
        try:
            plt.show()
        except:
            print("Could not display plot interactively")
        
        plt.close()

    

def do_ranking(model_paths, metrics):
    #function to help ranking the models based on the metrics
    inds=[3,2,4,7,1,10,-1,-2,5] #List of metrics indices to pick the correct score from the scores tensor
    dm=['S', 'MAE', 'E', 'F', 'WF', 'MBA', 'BIoU', 'MSE', 'HCE']
    metr_sel_ind=[]
    for m in metrics:
        metr_sel_ind.append(inds[dm.index(m)])
    print("Ranking models", model_paths, "based on metrics: ", metrics, "(with indices: ", metr_sel_ind, ")...")
    #find the paths of original images, ground truth and model predictions
    pred_data_dir = []
    for i, model_path in enumerate(model_paths):
        pred_data_dir.append(sorted([os.path.join("e_preds", model_path, file_name) for file_name in os.listdir("e_preds/{}".format(model_path))]))
    to_delete=config.testsets.replace("test","train")
    gt_paths = sorted(glob(os.path.join(to_delete, 'gt', '*')))
    image_paths = sorted(glob(os.path.join(to_delete, 'im', '*')))
    print(config.testsets)
    #initialize the figure
    lun = min(10,len(gt_paths)) #minimum between 10 and the number of testing images
    lar = len(model_paths)+2
    for m in metrics:
        plt.figure(m,figsize=(5*lar, 6*lun))
    
    zs=list(zip(np.transpose(pred_data_dir), gt_paths, image_paths))
    scoresss=[]
    #for each image compute the scores
    for im_ind, (p, g, m) in enumerate(zs):
        #load the images
        image_pred = []
        scoress=[]
        for mod_ind in range(len(p)):
            # Evaluate model predictions against ground truth
            em, sm, fm, mae, mse, wfm, hce, mba, biou = evaluator(
                gt_paths=[g],
                pred_paths=[p[mod_ind]],
                metrics=dm,
                verbose=config.verbose_eval
            )

            scores = [
                    fm['curve'].max().round(3), wfm.round(3), mae.round(3), sm.round(3), em['curve'].mean().round(3), int(hce.round()), 
                    em['curve'].max().round(3), fm['curve'].mean().round(3), em['adp'].round(3), fm['adp'].round(3),
                    mba.round(3), biou['curve'].max().round(3),mse.round(3), biou['curve'].mean().round(3),
                    ]
            scoress.append(scores)
        scoresss.append(scoress) #tensor containing the scores of all the images for all the models and metrics

    #Compute the most interesting images based on the metrics, and keep only them
    rs=compute_interest(scoresss,metr_sel_ind,lun)
    print("The most interesting images are: ", rs)
    z=[]
    for r in rs:
        z.append(zs[r])

    #Loop through all the interesting images, the models and the metrics
    for im_ind, (p, g, m) in enumerate(z):
        image=Image.open(m)
        image_gt=Image.open(g)
        image_pred=[]
        for mod_ind in range(len(p)):
            print("Opening image at: ", p[mod_ind])
            image_pred.append(Image.open(p[mod_ind]))
        for metric in metrics:
            # Display
            plt.figure(metric)
            plt.subplot(lun, lar, lar*im_ind+1)
            plt.imshow(image)
            plt.axis('off')
            plt.title('Original',fontsize=20)

            plt.subplot(lun, lar, lar*im_ind+2)
            plt.imshow(image_gt)
            plt.axis('off')
            plt.title('Ground truth',fontsize=20)

            for i in range(len(p)):
                plt.subplot(lun, lar, lar*im_ind+3+i)
                plt.imshow(image_pred[i])
                plt.axis('off')
                vet=p[i].split("/")[1:-1]
                tit=" ".join(v for v in vet)

                #Pick the correct score from the scores tensor
                score=scoresss[rs[im_ind]][i][inds[dm.index(metric)]]
                #Display the score
                print("Model: ", tit, "Metric: ", metric, "Score: ", score)
                plt.title("Model "+tit+"\n"+f"{metric}: {score}",fontsize=20)
    # Save one figure for each metric
    for metric in metrics:
        plt.figure(metric)
        plt.tight_layout()
        output_file = os.path.join("e_results", f'comparison_{metric}.png')
        print(f"Saving comparison for {metric} to: {output_file}")
        plt.savefig(output_file, bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Run visualization or ranking based on the provided parameters.')
    parser.add_argument('--models', type=str, required=True, help='Path to the models')
    parser.add_argument('--metrics', type=str, help='Metrics to be used',
                        default=','.join(['S', 'MAE', 'E', 'F', 'WF', 'MBA', 'BIoU', 'MSE', 'HCE']))

    # Parse the arguments
    args = parser.parse_args()

    # Call the appropriate function based on the comparison flag
    do_ranking(args.models.split(','), args.metrics.split(','))
    do_visualization(args.models.split(','))