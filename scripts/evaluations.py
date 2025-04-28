import os
import argparse
from glob import glob
import prettytable as pt

from evaluation.metrics import evaluator
from config import Config
from typing import List

import datetime as dt
from datetime import datetime as dt

config = Config()


def get_scores(list_gt: List[str], list_pred: List[str]):
    """
    Takes the list of GT and preds files
    Computes the scores for the predictions
    Return a row to be added to the table
    """
    #evaluate the predictions
    em, sm, fm, mae, mse, wfm, hce, mba, biou, pa = evaluator(
        gt_paths=list_gt,
        pred_paths=list_pred,
        #metrics=args.metrics.split('+'), if we want display only few metrics
        verbose=config.verbose_eval
    )

    #create a list containing all the computed scores
    scores = [
        fm['curve'].max().round(3), wfm.round(3), mae.round(3), mse.round(3), sm.round(3), em['curve'].mean().round(3), int(hce.round()), 
        em['curve'].max().round(3), fm['curve'].mean().round(3), em['adp'].round(3), fm['adp'].round(3),
        mba.round(3), biou['curve'].max().round(3), biou['curve'].mean().round(3), pa.round(3)
    ]

    #format the scores
    for idx_score, score in enumerate(scores):
        scores[idx_score] = '.' + format(score, '.3f').split('.')[-1] if score < 1  else format(score, '<4')

    #create a list containing the scores
    return scores


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

    filename = os.path.join(args.save_dir, 'eval_{}.txt'.format(current_time))
    tb = pt.PrettyTable()
    tb.vertical_char = '&'
    tb.field_names = ["Model", "Test set", "maxFm", "wFmeasure", 'MAE', 'MSE', "Smeasure", "meanEm", "HCE", "maxEm", "meanFm", "adpEm", "adpFm", 'mBA', 'maxBIoU', 'meanBIoU', 'PixAcc']

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

        #check if the ground-truth and prediction folders have the same elements
        assert len(list_gt) == len(list_pred), "The folder {} is not matching to the corresponding ground-truth folder".format(prediction)

        #get the title of the row to be added to the table
        title = prediction.split('/')

        #get the scores
        scores = get_scores(list_gt, list_pred)

        #create a list containing the title and the scores
        record=title+scores

        #add the row to the table
        tb.add_row(record)

        #write the results to the file
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