import logging
import os
import torch
from torchvision import transforms
import numpy as np
import random
import cv2
from PIL import Image
import wandb

def init_wandb(config,args):
        wandb.init(
        # Set the project where this run will be logged
        project="BiRefNet finetuning",
        dir="..",
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=config.run_name,
        # Track hyperparameters and run metadata
        config={
        "learning_rate": config.lr,
        "architecture": "BiRefNet",
        "epochs": args.epochs,
        "batch_size": config.batch_size,
        "optimizer": config.optimizer,
        "train_set": args.train_set,
        "validation_set": args.validation_set,
        "save_last_epochs": args.save_last_epochs,
        "save_each_epochs": args.save_each_epochs,
        "finetune_last_epochs": config.fine_tune_last,
        "pixel loss lambdas": config.lambdas_pix_last,
        "pixel loss lambdas activated": config.lambdas_pix_last_activated,
        "bce_with_logits": config.bce_with_logits,
        "lr_warm_up_type": config.lr_warm_up_type,
        "lr_decay_epochs": config.lr_decay_epochs,
        "lr_decay_rate": config.lr_decay_rate
        })
        
        wandb.define_metric("Gradient norm")
        wandb.define_metric("Training Loss")
        wandb.define_metric("Validation Loss")
        wandb.define_metric("Learning Rate")
        wandb.define_metric("Gradient norm")
        wandb.define_metric("BCE loss training")
        wandb.define_metric("SSIM loss training")
        wandb.define_metric("MAE loss training")
        wandb.define_metric("IoU loss training")
        wandb.define_metric("GDT loss training")
        wandb.define_metric("BCE loss validation")
        wandb.define_metric("SSIM loss validation")
        wandb.define_metric("MAE loss validation")
        wandb.define_metric("IoU loss validation")
        wandb.define_metric("GDT loss validation")
        wandb.define_metric("Boundary IoU")
        wandb.define_metric("Pixel Accuracy")
        wandb.define_metric("Contour loss")

def get_lr_warm_up_scheduler(type, epochs, start_factor, end_factor, optimizer):
    if type == 'linear':
        print("Leraning rate warm up type set to linear")
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=start_factor,
            end_factor=end_factor,
            total_iters=epochs
        )
    elif type == 'cosine':
        print("Leraning rate warm up type set to cosine")
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=start_factor
        )
    else:
        print("Invalid learning rate warm up type, skipping it...")
        lr_scheduler = None
    return lr_scheduler
        

def path_to_image(path, size=(1024, 1024), color_type=['rgb', 'gray'][0]):
    if color_type.lower() == 'rgb':
        image = cv2.imread(path)
    elif color_type.lower() == 'gray':
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        print('Select the color_type to return, either to RGB or gray image.')
        return
    if size:
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    if color_type.lower() == 'rgb':
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert('RGB')
    else:
        image = Image.fromarray(image).convert('L')
    return image



def check_state_dict(state_dict, unwanted_prefixes=['module.', '_orig_mod.']):
    for k, v in list(state_dict.items()):
        prefix_length = 0
        for unwanted_prefix in unwanted_prefixes:
            if k[prefix_length:].startswith(unwanted_prefix):
                prefix_length += len(unwanted_prefix)
        state_dict[k[prefix_length:]] = state_dict.pop(k)
    return state_dict


def generate_smoothed_gt(gts):
    epsilon = 0.001
    new_gts = (1-epsilon)*gts+epsilon/2
    return new_gts


class Logger():
    def __init__(self, path="log.txt"):
        self.logger = logging.getLogger('BiRefNet')
        self.file_handler = logging.FileHandler(path, "w")
        self.stdout_handler = logging.StreamHandler()
        self.stdout_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.stdout_handler)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
    
    def info(self, txt):
        self.logger.info(txt)
    
    def close(self):
        self.file_handler.close()
        self.stdout_handler.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, path, filename="latest.pth"):
    torch.save(state, os.path.join(path, filename))


def save_tensor_img(tenor_im, path):
    im = tenor_im.cpu().clone()
    im = im.squeeze(0)
    tensor2pil = transforms.ToPILImage()
    im = tensor2pil(im)
    im.save(path)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
