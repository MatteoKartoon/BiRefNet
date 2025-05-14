import os
import argparse
from glob import glob
import prettytable as pt

from birefnet.evaluation.metrics import evaluator
from birefnet.config import Config
from typing import List

import datetime as dt
from datetime import datetime as dt

config = Config()


def get_scores(list_gt: List[str], list_pred: List[str]):
    """
    Takes the list of GT and preds files
    Computes the scores for the predictions for the active metrics
    Return a row to be added to the table
    """
    #evaluate the predictions
    em, sm, fm, mae, mse, wfm, hce, mba, biou, pa = evaluator(
        gts=list_gt,
        preds=list_pred,
        metrics=config.display_eval_metrics,
        verbose=config.verbose_eval
    )

    #create a list containing all the computed scores
    scores = {'S': sm, 'MAE': mae, 'E': em, 'F': fm, 'WF': wfm, 'MBA': mba, 'BIoU': biou, 'MSE': mse, 'HCE': hce, 'PA': pa}

    scores = {metric:value['curve'].mean().round(3) if metric in ['E','F','BIoU'] else int(hce.round()) if metric == 'HCE' else value.round(3) for metric, value in scores.items()}

    #create a list containing the active scores
    return [scores[metric] for metric in config.display_eval_metrics]

def get_field_names():
    """
    Returns the field names for the table
    """
    metric_names={
        'S': 'S measure',
        'MAE': 'Mean Absolute Error',
        'E': 'Mean E measure',
        'F': 'Mean F measure',
        'WF': 'Weighted F measure',
        'MBA': 'Mean Boundary Accuracy',
        'BIoU': 'Mean Boundary IoU',
        'MSE': 'Mean Squared Error',
        'HCE': 'HCE',
        'PA': 'Pixel Accuracy'
    }
    return ["Model", "Test set", "# test images", *[metric_names[metric] for metric in config.display_eval_metrics]]


def do_eval(args):
    """
    Takes the arguments from the command line (or the default values) and evaluates the predictions, saving results in a table
    """
    #if the directory where the results will be saved does not exist, create it
    os.makedirs(args.save_dir, exist_ok=True)
    
    #get a list of the prediction folders we want to evaluate
    args.predictions = args.pred_path.split('+')

    #create a file to save the results
    current_time = dt.now().strftime("%Y%m%d__%H%M")

    filename = f"{args.save_dir}/eval_{current_time}.txt"
    tb = pt.PrettyTable()
    tb.vertical_char = '&'
    tb.field_names = get_field_names()

    #variables to store the results for each testset
    model_results_dict={}

    #loop over the prediction folders
    for prediction in args.predictions:
        #get the ground-truth and prediction precise paths
        gt_pth = os.path.join(args.gt_root, prediction.split('/')[-1],"gt")
        pred_pth = os.path.join(args.pred_root, prediction)

        #check if the ground-truth and prediction folders exist
        assert os.path.exists(gt_pth), "Ground-truth path does not exist"
        assert os.path.exists(pred_pth), "Prediction path does not exist"

        #print information while computing
        print("Evaluating predictions for model: ", prediction)
        print("Ground-truth path: ", gt_pth)
        print("Prediction path: ", pred_pth)

        #get the list of files in the ground-truth and prediction folders
        list_gt = sorted([os.path.join(gt_pth, f) for f in os.listdir(gt_pth)])
        list_pred = sorted([os.path.join(pred_pth, f) for f in os.listdir(pred_pth)])
        images_number = len(list_gt)

        #check if the ground-truth and prediction folders have the same elements
        assert len(list_gt) == len(list_pred), "The folder {} is not matching to the corresponding ground-truth folder".format(prediction)

        #get the title of the row to be added to the table (model name, test set, test image number)
        title = prediction.split('/')
        title.append(images_number)

        #get the scores
        scores = get_scores(list_gt, list_pred)

        #add the results to the dictionary
        model_name=title[0]
        weighted_scores=[images_number*s for s in scores]
        if model_name in model_results_dict:
            new_images_number=model_results_dict[model_name][0]+images_number
            new_weighted_scores=[model_results_dict[model_name][1][i]+weighted_scores[i] for i in range(len(weighted_scores))]
            model_results_dict[model_name]=(new_images_number,new_weighted_scores)
        else:
            model_results_dict[model_name] = (images_number,weighted_scores)

        #create a list containing the title and the scores
        formatted_scores=[f".{f'{score:.3f}'.split('.')[-1]}" if score < 1 else f"{score:<4}" for score in scores]
        record=title+formatted_scores

        #add the row to the table
        tb.add_row(record)

        #write the results to the file
        with open(filename, 'w+') as file_to_write:
            file_to_write.write(str(tb)+'\n')
    tb.add_divider()

    #compute the average scores for each testset and add to the table
    for model_name, (images_number, weighted_scores) in model_results_dict.items():
        #compute the average scores and format them
        average_scores=[s/images_number for s in weighted_scores]
        average_scores=[f".{f'{score:.3f}'.split('.')[-1]}" if score < 1 else f"{score:<4}" for score in average_scores]

        #create a list containing the title and the average scores
        title=[model_name, "Model average", images_number]
        record=title+average_scores

        #add the row to the table
        tb.add_row(record)
        with open(filename, 'w+') as file_to_write:
            file_to_write.write(str(tb)+'\n')

    #confirm that the evaluation is completed
    print("Evaluation completed. Results saved in {}".format(filename))



if __name__ == '__main__':
    # set parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pred_path', type=str, help='pred path')
    parser.add_argument(
        '--gt_root', type=str, help='ground-truth root',
        default=os.path.join(config.data_root_dir, config.task))
    parser.add_argument(
        '--pred_root', type=str, help='prediction root',
        default='../e_preds')
    parser.add_argument(
        '--save_dir', type=str, help='candidate competitors',
        default='../e_results')
    parser.add_argument(
        '--metrics', type=str, help='candidate competitors',
        default='+'.join(['S', 'MAE', 'E', 'F', 'WF', 'MBA', 'BIoU', 'MSE', 'HCE','PA'])
    )
    args = parser.parse_args()

    # start computing
    do_eval(args)