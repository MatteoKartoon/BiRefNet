import os
import argparse
from glob import glob
from tqdm import tqdm
import cv2
import torch
import re

from birefnet.dataset import MyData
from birefnet.models.birefnet import BiRefNet, BiRefNetC2F
from birefnet.utils import save_tensor_img, check_state_dict
from birefnet.config import Config


config = Config()


def inference(model, data_loader_test, pred_root, testset, device=0):
    model_training = model.training
    if model_training:
        model.eval()
    model.half()
    assert validate_format(args.ckpt_path), "Checkpoint path must be in the format dddddddd__dddd/epoch_ddd.pth where d is a digit"
    #get epoch from ckpt path name
    epoch = args.ckpt_path.split('_')[-1].split('.')[0]
    #get the tr
    training_date=args.ckpt_path.split('/')[0]
    #create the directory to save the predictions
    saving_dir=f"{pred_root}/{training_date}__epoch{epoch}/{testset}"
    print(f"Saving prediction in {saving_dir}...")

    for batch in tqdm(data_loader_test, total=len(data_loader_test)) if 1 or config.verbose_eval else data_loader_test:
        inputs = batch[0].to(device).half()
        # gts = batch[1].to(device)
        label_paths = batch[-1]
        with torch.no_grad():
            scaled_preds = model(inputs)[-1].sigmoid()

        #create the directory to save the predictions
        os.makedirs(saving_dir, exist_ok=True)

        for idx_sample in range(scaled_preds.shape[0]):
            res = torch.nn.functional.interpolate(
                scaled_preds[idx_sample].unsqueeze(0),
                size=cv2.imread(label_paths[idx_sample], cv2.IMREAD_GRAYSCALE).shape[:2],
                mode='bilinear',
                align_corners=True
            )
            #prediction image path
            computed_prediction_name=label_paths[idx_sample].replace('\\', '/').split('/')[-1]
            complete_saving_path=os.path.join(saving_dir, computed_prediction_name) # test set dir + file name
            save_tensor_img(res, complete_saving_path)
    if model_training:
        model.train()
    return None


def validate_format(value):
    """
    Validates if a string matches the format dddddddd__dddd/epoch_ddd.pth where d is a digit
    Returns True if the format is valid, False otherwise
    """
    pattern = r'^\d{8}__\d{4}/epoch_\d{3}\.pth$'
    return bool(re.match(pattern, value))


def main(args):
    # Init model
    weights = os.path.join(args.ckpt_folder, args.ckpt_path)
    device = config.device
    model = BiRefNet(bb_pretrained=False)
        
    #start testing
    for testset in args.testsets.split('+'):
        testset_pth=os.path.join(config.data_root_dir,"fine_tuning",testset)
        print('>>>> Testset: {}'.format(testset_pth))
        data_loader_test = torch.utils.data.DataLoader(
            dataset=MyData(testset_pth, image_size=config.size, is_train=False),
            batch_size=config.batch_size_test, shuffle=False, num_workers=config.num_workers, pin_memory=True
        )
        
        print('\tInferencing {}...'.format(weights))
        state_dict = torch.load(weights, map_location='cpu', weights_only=True)
        state_dict = check_state_dict(state_dict)
        model.load_state_dict(state_dict)
        model = model.to(device)
        inference(
            model, data_loader_test=data_loader_test, pred_root=args.pred_root,
            testset=testset, device=config.device
        )


if __name__ == '__main__':
    # Parameter from command line
    parser = argparse.ArgumentParser(description='')
    
    parser.add_argument('--ckpt_folder', default="../ckpt/fine_tuning", type=str, help='model folder')
    parser.add_argument('--ckpt_path', type=str, help='which checkpoint to use')
    parser.add_argument('--pred_root', default='../e_preds', type=str, help='Output folder')
    parser.add_argument('--testsets',type=str,help="which test set do inference on")

    args = parser.parse_args()

    if config.precisionHigh:
        torch.set_float32_matmul_precision('high')
    main(args)
