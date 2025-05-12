import os
import datetime
from contextlib import nullcontext
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime as dt
import wandb
from typing import List
import ast

if tuple(map(int, torch.__version__.split('+')[0].split(".")[:3])) >= (2, 5, 0):
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from birefnet.config import Config
from birefnet.loss import PixLoss, ClsLoss
from birefnet.dataset import MyData
from birefnet.models.birefnet import BiRefNet, BiRefNetC2F
from birefnet.evaluation.metrics import evaluator
from birefnet.utils import Logger, AverageMeter, set_seed, check_state_dict, init_wandb, get_lr_warm_up_scheduler

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

weights_dir = '../../../../weights/cv'

parser = argparse.ArgumentParser(description='')
parser.add_argument('--resume', default=None, type=str, help='path to latest checkpoint')
parser.add_argument('--epochs', default=120, type=int)
parser.add_argument('--ckpt_dir', default='ckpt/tmp', help='Temporary folder')
parser.add_argument('--dist', default=False, type=lambda x: x == 'True')
parser.add_argument('--use_accelerate', action='store_true', help='`accelerate launch --multi_gpu train.py --use_accelerate`. Use accelerate for training, good for FP16/BF16/...')
parser.add_argument('--train_set', type=str, help='Training set')
parser.add_argument('--validation_set', type=str, help='Validation set')
parser.add_argument('--save_last_epochs', default=10, type=int)
parser.add_argument('--save_each_epochs', default=2, type=int)
#Parameters to be passed to the config class
parser.add_argument('--learning_rate', default=None, type=float)
parser.add_argument('--bce_with_logits', default=None, type=bool)
parser.add_argument('--lambdas_pix_last', default=None, type=ast.literal_eval)
parser.add_argument('--lambdas_pix_last_activated', default=None, type=ast.literal_eval)
parser.add_argument('--run_name', default=None, type=str)

args = parser.parse_args()

if args.use_accelerate:
    from accelerate import Accelerator, utils
    mixed_precision = 'bf16'
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=1,
        kwargs_handlers=[
            utils.InitProcessGroupKwargs(backend="nccl", timeout=datetime.timedelta(seconds=3600*10)),
            utils.DistributedDataParallelKwargs(find_unused_parameters=True),
            utils.GradScalerKwargs(backoff_factor=0.5)],
    )
    args.dist = False

config = Config(
    learning_rate=args.learning_rate,
    bce_with_logits=args.bce_with_logits,
    lambdas_pix_last=args.lambdas_pix_last,
    lambdas_pix_last_activated=args.lambdas_pix_last_activated,
    run_name=args.run_name
)
if config.rand_seed:
    set_seed(config.rand_seed)

if accelerator.is_main_process:
    init_wandb(config,args)

# DDP
to_be_distributed = args.dist
if to_be_distributed:
    init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600*10))
    device = int(os.environ["LOCAL_RANK"])
else:
    if args.use_accelerate:
        device = accelerator.device
    else:
        device = config.device

epoch_st = 1

# Create a folder inside ckpt_dir based on date with format yyyymmdd__hhmm
current_time = dt.now().strftime("%Y%m%d__%H%M")
args.ckpt_dir = os.path.join(args.ckpt_dir, current_time)

# make dir for ckpt
os.makedirs(args.ckpt_dir, exist_ok=True)

# Init log file
logger = Logger(os.path.join(args.ckpt_dir, "log.txt"))
logger_loss_idx = 1

# log model and optimizer params
# logger.info("Model details:"); logger.info(model)
# if args.use_accelerate and accelerator.mixed_precision != 'no':
#     config.compile = False
logger.info("Task: {}".format(config.task))
logger.info("datasets: load_all={}, compile={}.".format(config.load_all, config.compile))
logger.info("Other hyperparameters:"); logger.info(args)
print('batch size:', config.batch_size)

from birefnet.dataset import custom_collate_fn

def prepare_dataloader(dataset: torch.utils.data.Dataset, batch_size: int, to_be_distributed=False, is_train=True):
    # Prepare dataloaders
    if to_be_distributed:
        return torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, num_workers=min(config.num_workers, batch_size), pin_memory=True,
            shuffle=False, sampler=DistributedSampler(dataset), drop_last=True, collate_fn=custom_collate_fn if is_train and config.dynamic_size != (0, 0) else None
        )
    else:
        return torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, num_workers=min(config.num_workers, batch_size), pin_memory=True,
            shuffle=is_train, sampler=None, drop_last=True, collate_fn=custom_collate_fn if is_train and config.dynamic_size != (0, 0) else None
        )


def get_scores(list_gt: List[str], list_pred: List[str]):
    """
    Takes the list of GT and preds files
    Computes the scores for the predictions for the active metrics
    Return a dictionary containing the scores for the active metrics
    """
    #evaluate the predictions
    em, sm, fm, mae, mse, wfm, hce, mba, biou, pa = evaluator(
        gt_paths=list_gt,
        pred_paths=list_pred,
        metrics=config.display_eval_metrics,
        verbose=config.verbose_eval
    )

    #create a list containing all the computed scores
    scores = {'S': sm, 'MAE': mae, 'E': em, 'F': fm, 'WF': wfm, 'MBA': mba, 'BIoU': biou, 'MSE': mse, 'HCE': hce, 'PA': pa}

    scores = {metric:value['curve'].mean().round(3) if metric in ['E','F','BIoU'] else int(hce.round()) if metric == 'HCE' else value.round(3) for metric, value in scores.items()}

    #create a list containing the active scores
    return {metric:score for metric,score in scores.items() if metric in config.display_eval_metrics}


def init_data_loaders(to_be_distributed):
    # Prepare datasets
    training_set = os.path.join(config.data_root_dir, config.task, args.train_set)
    validation_set = os.path.join(config.data_root_dir, config.task, args.validation_set)
    train_loader = prepare_dataloader(
        MyData(datasets=training_set, image_size=config.size, is_train=True),
        config.batch_size, to_be_distributed=to_be_distributed, is_train=True
    )

    validation_loader = prepare_dataloader(
        MyData(datasets=validation_set, image_size=config.size, is_train=True),
        config.batch_size, to_be_distributed=to_be_distributed, is_train=True
    )
    print(len(train_loader), "batches of train dataloader {} have been created.".format(training_set))
    print(len(validation_loader), "batches of validation dataloader {} have been created.".format(validation_set))
    return train_loader, validation_loader


def init_models_optimizers(epochs, to_be_distributed):
    # Init models
    if config.model == 'BiRefNet':
        model = BiRefNet(bb_pretrained=True and not os.path.isfile(str(args.resume)))
    elif config.model == 'BiRefNetC2F':
        model = BiRefNetC2F(bb_pretrained=True and not os.path.isfile(str(args.resume)))
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            state_dict = torch.load(args.resume, map_location='cpu', weights_only=True)
            state_dict = check_state_dict(state_dict)
            model.load_state_dict(state_dict)
            global epoch_st
            epoch_st = int(args.resume.rstrip('.pth').split('epoch_')[-1]) + 1
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))
    if not args.use_accelerate:
        if to_be_distributed:
            model = model.to(device)
            model = DDP(model, device_ids=[device])
        else:
            model = model.to(device)
    if config.compile:
        model = torch.compile(model, mode=['default', 'reduce-overhead', 'max-autotune'][0])
    if config.precisionHigh:
        torch.set_float32_matmul_precision('high')

    # Setting optimizer
    if config.optimizer == 'AdamW':
        optimizer = optim.AdamW(params=model.parameters(), lr=config.lr, weight_decay=1e-2)
        for param_group in optimizer.param_groups:
            param_group['initial_lr'] = config.lr


    # Create a lr warmup for first 10 epochs
    wu_epochs = 10
    warmup_scheduler = get_lr_warm_up_scheduler(config.lr_warm_up_type, wu_epochs, 1e-20, 1.0, optimizer)
    
    # Main scheduler after warmup
    main_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config.lr_decay_epochs,
        gamma=config.lr_decay_rate
    )
    
    if warmup_scheduler != None:
        # Combine both schedulers
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[wu_epochs]
        )
    else:
        lr_scheduler = main_scheduler

    return model, optimizer, lr_scheduler


class Trainer:
    def __init__(
        self, data_loaders, model_opt_lrsch,
    ):
        self.model, self.optimizer, self.lr_scheduler = model_opt_lrsch
        self.train_loader,self.validation_loader = data_loaders
        if args.use_accelerate:
            self.train_loader, self.validation_loader, self.model, self.optimizer = accelerator.prepare(
                self.train_loader, self.validation_loader, self.model, self.optimizer
            )

        # Setting Losses
        self.pix_loss = PixLoss(config)
        self.cls_loss = ClsLoss()
        
        if config.out_ref:
            if config.bce_with_logits:
                self.criterion_gdt = nn.BCEWithLogitsLoss()
            else:
                self.criterion_gdt = nn.BCELoss()

        # Others
        self.loss_log = AverageMeter()
        self.val_loss_log = AverageMeter()
        
        assert args.save_last_epochs >= 0, "save_last_epochs must be greater than 0"
        assert config.finetune_last_epochs <= 0, "finetune_last_epochs must be less than 0"
        self.save_last_epochs_start = args.epochs - args.save_last_epochs
        self.finetune_last_epochs_start = args.epochs + config.finetune_last_epochs

        self.last_grad_norm = 0

    def _get_loss_key(self,epoch):
        """
        We change the loss function in the last epochs so we change the name of the dictionary key in order to not confuse the two
        """
        return "loss_pix_rescaled" if epoch > self.finetune_last_epochs_start else "loss_pix"

    def iteration_over_batches_validation(self, epoch, info_progress=None, step_idx=None, training_result=None):
        #Loop over the batches
        loss_components_dict={k:0 for k in self.loss_components_train.keys()}
        validation_metrics_dict = {k:0 for k in config.display_eval_metrics}

        #Compute the sum of the losses over the validation set batches
        for batch_idx, batch in enumerate(self.validation_loader):
            self._batch(batch, batch_idx, epoch, validation=True)
            # Logger
            for loss_name, loss_value in self.loss_components_validation.items():
                loss_components_dict[loss_name]+=loss_value
            for metric, score in self.validation_metrics.items():
                validation_metrics_dict[metric]+=score

        #Compute the average of the losses over the validation set
        loss_components_dict={loss_name:loss_accumulated_value/len(self.validation_loader) for loss_name, loss_accumulated_value in loss_components_dict.items()}
        validation_metrics_dict = {metric:score/len(self.validation_loader) for metric, score in validation_metrics_dict.items()}

        #For each loss type, compute the average between the devices
        average_total_loss=sum(loss_components_dict.values())
        loss_general_value=self.average_between_devices(average_total_loss)
        loss_components_dict={loss_name:self.average_between_devices(loss_accumulated_value) for loss_name, loss_accumulated_value in loss_components_dict.items()}
        validation_metrics_dict = {metric:self.average_between_devices(score) for metric, score in validation_metrics_dict.items()}
        accelerator.wait_for_everyone()

        #add to the print string and log on wandb
        if accelerator.is_main_process:
            info_loss = f'Validation Losses {loss_general_value}'
            logger.info(' '.join((info_progress, info_loss)))
            wandb.log({"Validation Loss": loss_general_value,
                       "Training Loss": training_result,
                       "Learning Rate": self.lr_scheduler.get_last_lr()[0],
                       "Gradient Norm": self.last_grad_norm,
                       "BCE loss validation": loss_components_dict['bce'],
                       "SSIM loss validation": loss_components_dict['ssim'],
                       "MAE loss validation": loss_components_dict['mae'],
                       "IoU loss validation": loss_components_dict['iou'],
                       "GDT loss validation": loss_components_dict['gdt'],
                       "BCE loss training": self.loss_components_train['bce'],
                       "SSIM loss training": self.loss_components_train['ssim'],
                       "MAE loss training": self.loss_components_train['mae'],
                       "IoU loss training": self.loss_components_train['iou'],
                       "GDT loss training": self.loss_components_train['gdt'],
                       "Boundary IoU": validation_metrics_dict['BIoU'],
                       "Pixel Accuracy": validation_metrics_dict['PA'],
                       },step=step_idx)
        accelerator.wait_for_everyone()

    def iteration_over_batches_train(self, epoch):
        #Loop over the training batches
        for batch_idx, batch in enumerate(self.train_loader):
            step_idx=batch_idx+len(self.train_loader)*(epoch-config.start_epoch)+1
            self._batch(batch, batch_idx, epoch,validation=False)
            # Logger
            if batch_idx % config.log_each_steps == 0:
                info_progress = f'Epoch[{epoch}/{args.epochs}] Iter[{batch_idx}/{len(self.train_loader)}].'
                loss_general_value=self.average_between_devices(self.loss_dict_train[self._get_loss_key(epoch)])
                self.loss_components_train={loss_name:self.average_between_devices(loss_value) for loss_name, loss_value in self.loss_components_train.items()}
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    info_loss = f'Training Losses {loss_general_value}'
                    logger.info(' '.join((info_progress, info_loss)))
                self.iteration_over_batches_validation(epoch, info_progress, step_idx, loss_general_value)

    def average_between_devices(self, values: float):
        #gather values from all processes
        values_tensor = torch.tensor([values], device=accelerator.device)
        all_values = accelerator.gather(values_tensor)
        # Compute average loss on main process
        if accelerator.is_main_process:
            return all_values.mean().item()

    def epoch_final_logs(self,epoch, log_losses, log_task="Training"):
        #Print the final epoch logs
        info_loss = f'@==Final== Epoch[{epoch}/{args.epochs}]  {log_task} Loss Device {accelerator.device}: {log_losses.avg}'
        logger.info(info_loss)
        avg_loss = self.average_between_devices(log_losses.avg)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            logger.info(f'@==Final== Epoch[{epoch}/{args.epochs}]  Average {log_task} Loss: {avg_loss}')
        # Synchronize before next steps
        accelerator.wait_for_everyone()

    def _batch(self, batch, batch_idx, epoch, validation=False):
        if args.use_accelerate:
            inputs = batch[0]#.to(device)
            gts = batch[1]#.to(device)
            class_labels = batch[2]#.to(device)
        else:
            inputs = batch[0].to(device)
            gts = batch[1].to(device)
            class_labels = batch[2].to(device)
        self.optimizer.zero_grad()
        scaled_preds, class_preds_lst = self.model(inputs)
        loss_dict=self.loss_dict_validation if validation else self.loss_dict_train
        if config.out_ref:
            # Only unpack if in training mode and out_ref is enabled
            (outs_gdt_pred, outs_gdt_label), scaled_preds = scaled_preds
            for _idx, (_gdt_pred, _gdt_label) in enumerate(zip(outs_gdt_pred, outs_gdt_label)):
                _gdt_pred = nn.functional.interpolate(_gdt_pred, size=_gdt_label.shape[2:], mode='bilinear', align_corners=True)
                if not config.bce_with_logits:
                    _gdt_pred = _gdt_pred.sigmoid()
                _gdt_label = _gdt_label.sigmoid()
                loss_gdt = self.criterion_gdt(_gdt_pred, _gdt_label) if _idx == 0 else self.criterion_gdt(_gdt_pred, _gdt_label) + loss_gdt
        if None in class_preds_lst:
            loss_cls = 0.
        else:
            loss_cls = self.cls_loss(class_preds_lst, class_labels) * 1.0
            loss_dict['loss_cls'] = loss_cls.item()
        
        # Loss
        loss_pix, loss_components = self.pix_loss(scaled_preds, torch.clamp(gts, 0, 1))
        loss_components['gdt'] = loss_gdt.item()
        if validation:
            self.loss_components_validation=loss_components
        else:
            self.loss_components_train=loss_components
        loss_dict[self._get_loss_key(epoch)] = loss_pix.item()+loss_gdt.item()

        # since there may be several losses for sal, the lambdas for them (lambdas_pix) are inside the loss.py
        loss = loss_pix + loss_cls
        if config.out_ref:
            loss = loss + loss_gdt

        if not validation:
            self.loss_log.update(loss.item(), inputs.size(0))
            if args.use_accelerate:
                loss = loss / accelerator.gradient_accumulation_steps
                accelerator.backward(loss)
            else:
                loss.backward()
            self.optimizer.step()

            # Print gradient norm to monitor training
            if batch_idx % config.log_each_steps == 0:
                total_norm = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                print("Gradient norm:", total_norm)
                self.last_grad_norm = total_norm
                if accelerator.is_main_process:
                    gt_image=wandb.Image(gts[0], caption="Ground Truth")
                    res = torch.nn.functional.interpolate(
                        scaled_preds[3][0].sigmoid().unsqueeze(0),
                        size=gts[0].shape[1:],
                        mode='bilinear',
                        align_corners=True
                    )
                    metric_scores = get_scores(gts[0], res[0])
                    caption = f"Predicted\nPixel Accuracy: {metric_scores['PA']}, Boundary IoU: {metric_scores['BIoU']}"
                    pred_image=wandb.Image(res, caption=caption)
                    wandb.log({"GT and prediction": [gt_image, pred_image]})
        else:
            gt_list = []
            pred_list = []
            for i in range(gts.size(0)):
                res = torch.nn.functional.interpolate(
                        scaled_preds[3][i].sigmoid().unsqueeze(0),
                        size=gts[i].shape[1:],
                        mode='bilinear',
                        align_corners=True
                )
                gt_list.append(gts[i][0])
                pred_list.append(res[0][0])
            self.validation_metrics = get_scores(gt_list, pred_list)
            self.val_loss_log.update(loss.item(), inputs.size(0))

    def train_epoch(self, epoch):
        global logger_loss_idx
        self.model.train()
        self.loss_dict_train = {}
        self.loss_dict_validation = {}
        if epoch == self.finetune_last_epochs_start:
            self.pix_loss.lambdas_pix_last['bce'] *= 0
            self.pix_loss.lambdas_pix_last['ssim'] *= 1
            self.pix_loss.lambdas_pix_last['iou'] *= 0.5
            self.pix_loss.lambdas_pix_last['mae'] *= 0.9
            print("Loss computation updated for the last epochs")
            self.loss_log = AverageMeter()
            self.val_loss_log = AverageMeter()

        self.iteration_over_batches_train(epoch)
        self.epoch_final_logs(epoch, self.loss_log, log_task="Training")
        
        self.lr_scheduler.step()

        self.epoch_final_logs(epoch, self.val_loss_log, log_task="Validation")

        return self.loss_log.avg, self.val_loss_log.avg


def main():
    save_to_cpu = True #otherwise we have crashing saving the checkpoint

    trainer = Trainer(
        data_loaders=init_data_loaders(to_be_distributed),
        model_opt_lrsch=init_models_optimizers(args.epochs, to_be_distributed)
    )

    for epoch in range(epoch_st, args.epochs+1):
        train_loss, val_loss = trainer.train_epoch(epoch)
        # Save checkpoint
        # DDP
        if epoch >= args.epochs - args.save_last_epochs and epoch % args.save_each_epochs == 0:
            if save_to_cpu:
                state_dict = {k: v.cpu() for k, v in trainer.model.state_dict().items()}
            # default behavior
            else:
                if args.use_accelerate:
                    if mixed_precision == 'fp16':
                        state_dict = {k: v.half() for k, v in trainer.model.state_dict().items()}
                else:
                    state_dict = trainer.model.module.state_dict() if to_be_distributed else trainer.model.state_dict()
            torch.save(state_dict, os.path.join(args.ckpt_dir, 'epoch_{}.pth'.format(epoch)))
    if to_be_distributed:
        destroy_process_group()


if __name__ == '__main__':
    main()