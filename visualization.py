import argparse
from PIL import Image
import os
import argparse
from glob import glob
import prettytable as pt
import matplotlib
import random

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from evaluation.metrics import evaluator
from config import Config

config = Config()

def pt_green_bg(original, mask):
    # Open the original image and the mask
    original_img = Image.open(original).convert("RGBA")
    mask_img = Image.open(mask).convert("L")

    # Create a new image with the same size as the original, filled with the green background color
    green_bg = Image.new("RGBA", original_img.size, (0, 255, 17, 255))

    # Composite the original image onto the green background using the mask
    result_img = Image.composite(original_img, green_bg, mask_img)

    return result_img


def do_visualization(model_paths):
    dm=['S', 'MAE', 'E', 'F', 'WF', 'MBA', 'BIoU', 'MSE', 'HCE']
    gt_paths = sorted(glob(os.path.join(config.testsets, 'gt', '*')))
    image_paths = sorted(glob(os.path.join(config.testsets, 'im', '*')))
    for model_path in model_paths:
        print("Visualizing model results: ", model_path)
        # Load the model predictions
        pred_data_dir = [os.path.join("e_preds", model_path, file_name) for file_name in os.listdir("e_preds/{}".format(model_path))]
        
        pred_data_dir = sorted(pred_data_dir)
        print(pred_data_dir)
        lun = len(pred_data_dir)
        plt.figure(figsize=(25, 8*lun))
        
        # Evaluate model predictions against ground truth
        rs=random.sample(range(0,len(gt_paths)),min(10,len(gt_paths)))
        print(rs)
        zs=list(zip(np.transpose(pred_data_dir), gt_paths, image_paths))
        z=[]
        for r in rs:
            z.append(zs[r])
        for i, (pred_path, gt_path, image_path) in enumerate(zip(pred_data_dir, gt_paths, image_paths)):
            print('\t', 'Evaluating prediction: {} against ground truth: {}'.format(pred_path, gt_path))
            em, sm, fm, mae, mse, wfm, hce, mba, biou = evaluator(
                gt_paths=[gt_path],
                pred_paths=[pred_path],
                metrics=dm,
                verbose=config.verbose_eval
            )

            scores = [
                    fm['curve'].max().round(3), wfm.round(3), mae.round(3), sm.round(3), em['curve'].mean().round(3), int(hce.round()), 
                    em['curve'].max().round(3), fm['curve'].mean().round(3), em['adp'].round(3), fm['adp'].round(3),
                    mba.round(3), biou['curve'].max().round(3),mse.round(3), biou['curve'].mean().round(3),
            ]
            print(scores)
            # Display
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
            plt.text(0.5, 0.5, f"maxFm: {scores[0]}\nwFmeasure: {scores[1]}\nMAE: {scores[2]}\nMSE: {scores[3]}\nSmeasure: {scores[4]}\n"
                    f"meanEm: {scores[5]}\nHCE: {scores[6]}\nmaxEm: {scores[7]}\nmeanFm: {scores[8]}\n"
                    f"adpEm: {scores[9]}\nadpFm: {scores[10]}\nmBA: {scores[11]}\nmaxBIoU: {scores[12]}\nmeanBIoU: {scores[13]}",
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
    inds=[3,2,4,7,1,10,-1,-2,5]
    dm=['S', 'MAE', 'E', 'F', 'WF', 'MBA', 'BIoU', 'MSE', 'HCE']
    print("Ranking models", model_paths, "based on metrics: ", metrics)
    #find the paths of original images, ground truth and model predictions
    pred_data_dir = []
    for i, model_path in enumerate(model_paths):
        pred_data_dir.append(sorted([os.path.join("e_preds", model_path, file_name) for file_name in os.listdir("e_preds/{}".format(model_path))]))

    gt_paths = sorted(glob(os.path.join(config.testsets, 'gt', '*')))
    image_paths = sorted(glob(os.path.join(config.testsets, 'im', '*')))
    #initialize the figure
    lun = min(16//len(metrics),9)*len(metrics)
    lar = len(model_paths)+2
    plt.figure(figsize=(5*lar, 6*lun))
    #iterate over some random images of the testset
    magic_ind=0
    rs=random.sample(range(0,len(gt_paths)),min(16//len(metrics),9))
    print(rs)
    zs=list(zip(np.transpose(pred_data_dir), gt_paths, image_paths))
    z=[]
    for r in rs:
        z.append(zs[r])
    for p, g, m in z:
        #load the images
        image_pred = []
        scoress=[]
        for i in range(len(p)):
            print("Opening image at: ", p[i])
            image_pred.append(Image.open(p[i]))
            # Evaluate model predictions against ground truth
            em, sm, fm, mae, mse, wfm, hce, mba, biou = evaluator(
                gt_paths=[g],
                pred_paths=[p[i]],
                metrics=dm,
                verbose=config.verbose_eval
            )

            scores = [
                    fm['curve'].max().round(3), wfm.round(3), mae.round(3), sm.round(3), em['curve'].mean().round(3), int(hce.round()), 
                    em['curve'].max().round(3), fm['curve'].mean().round(3), em['adp'].round(3), fm['adp'].round(3),
                    mba.round(3), biou['curve'].max().round(3),mse.round(3), biou['curve'].mean().round(3),
                    ]
            scoress.append(scores)

        image=Image.open(m)
        image_gt=Image.open(g)

        for metric in metrics:
            # Display
            # Set a title for each line of images
            #plt.text(0.5, 1 - magic_ind * 0.05, f"Metric: {metric}", fontsize=20, ha='center', transform=plt.gcf().transFigure)
            
            plt.subplot(lun, lar, lar*magic_ind+1)
            plt.imshow(image)
            plt.axis('off')
            plt.title('Original',fontsize=20)

            plt.subplot(lun, lar, lar*magic_ind+2)
            plt.imshow(image_gt)
            plt.axis('off')
            plt.title('Ground truth',fontsize=20)

            for i in range(len(p)):
                plt.subplot(lun, lar, lar*magic_ind+3+i)
                plt.imshow(image_pred[i])
                plt.axis('off')
                vet=p[i].split("/")[1:-1]
                tit=" ".join(v for v in vet)
                
                score=scoress[i][inds[dm.index(metric)]]

                print("Model: ", tit, "Metric: ", metric, "Score: ", score)
                plt.title("Model "+tit+"\n"+f"{metric}: {score}",fontsize=20)
            magic_ind+=1
    plt.tight_layout()
    # Save the figure
    output_file = os.path.join("e_results", f'comparison.png')
    print(f"Saving comparison to: {output_file}")
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    
    # Try to display (will work in interactive environments)
    try:
        plt.show()
    except:
        print("Could not display plot interactively")
    
    plt.close()


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