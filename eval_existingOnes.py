import os
import argparse
from glob import glob
import prettytable as pt

from evaluation.metrics import evaluator
from config import Config


config = Config()


def do_eval(args):
    # evaluation for whole dataset
    # dataset first in evaluation
    
    print('#' * 20, args.data_lst, '#' * 20)
    filename = os.path.join(args.save_dir, '{}_eval.txt'.format(args.data_lst))
    tb = pt.PrettyTable()
    tb.vertical_char = '&'
    tb.field_names = ["Training date", "Epoch", "maxFm", "wFmeasure", 'MAE', 'MSE', "Smeasure", "meanEm", "HCE", "maxEm", "meanFm", "adpEm", "adpFm", 'mBA', 'maxBIoU', 'meanBIoU']
        
    for model_name in args.model_lst: #loop over the folder given as input
        #model predictions loading
        pred_data_dir = [os.path.join("e_preds", model_name, file_name) for file_name in os.listdir("e_preds/{}".format(model_name))] #list of the files in the folder e_preds/training_date/training_epoch
        gt_paths = sorted(glob(os.path.join(config.testsets, 'gt', '*')))
        pred_data_dir=sorted(pred_data_dir)
        print(pred_data_dir)
        #model evaluation
        print('\t', 'Evaluating model: {}...'.format(model_name))
        pred_paths = [os.path.join(".",p) for p in pred_data_dir]
        em, sm, fm, mae, mse, wfm, hce, mba, biou = evaluator(
            gt_paths=gt_paths,
            pred_paths=pred_paths,
            metrics=args.metrics.split('+'),
            verbose=config.verbose_eval
        )

        scores = [
            fm['curve'].max().round(3), wfm.round(3), mae.round(3), mse.round(3), sm.round(3), em['curve'].mean().round(3), int(hce.round()), 
            em['curve'].max().round(3), fm['curve'].mean().round(3), em['adp'].round(3), fm['adp'].round(3),
            mba.round(3), biou['curve'].max().round(3), biou['curve'].mean().round(3),
        ]

        for idx_score, score in enumerate(scores):
            scores[idx_score] = '.' + format(score, '.3f').split('.')[-1] if score <= 1  else format(score, '<4')
        title = model_name.split('/')
        print(title)
        if len(title) > 1:
            records = title + scores
        else:
            records=title+["--"]+scores
        tb.add_row(records)
        # Write results after every check.
        with open(filename, 'w+') as file_to_write:
            file_to_write.write(str(tb)+'\n')
    print(tb)


if __name__ == '__main__':
    # set parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ckpt_path', type=str, help='ckpt path')
    parser.add_argument(
        '--gt_root', type=str, help='ground-truth root',
        default=os.path.join(config.data_root_dir, config.task))
    parser.add_argument(
        '--pred_root', type=str, help='prediction root',
        default='./e_preds')
    parser.add_argument(
        '--data_lst', type=str, help='test dataset',
        default='fine_tuning')
    parser.add_argument(
        '--save_dir', type=str, help='candidate competitors',
        default='./e_results')
    parser.add_argument(
        '--check_integrity', type=bool, help='whether to check the file integrity',
        default=False)
    parser.add_argument(
        '--metrics', type=str, help='candidate competitors',
        default='+'.join(['S', 'MAE', 'E', 'F', 'WF', 'MBA', 'BIoU', 'MSE', 'HCE'][:100 if 'DIS5K' in config.task else -1]))
    args = parser.parse_args()
    args.metrics = '+'.join(['S', 'MAE', 'E', 'F', 'WF', 'MBA', 'BIoU', 'MSE', 'HCE'][:100 if sum(['DIS-' in _data for _data in args.data_lst.split('+')]) else -1])

    os.makedirs(args.save_dir, exist_ok=True)
    
    args.model_lst = args.ckpt_path.split(',')

    # check the integrity of each candidates
    if args.check_integrity:
        for model_name in args.model_lst:
            gt_pth = os.path.join(args.gt_root, args.data_lst)
            pred_pth = os.path.join(args.pred_root, args.model_lst, args.data_lst)
            if not sorted(os.listdir(gt_pth)) == sorted(os.listdir(pred_pth)):
                print(len(sorted(os.listdir(gt_pth))), len(sorted(os.listdir(pred_pth))))
                print('The {} Dataset of {} Model is not matching to the ground-truth'.format(args.data_lst, args.model_lst))
    else:
        print('>>> skip check the integrity of each candidates')

    # start engine
    do_eval(args)
