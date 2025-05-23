import os
import math
from datetime import datetime as dt

#Set the default values for the parameters
DEFAULT_LR = 1e-5
DEFAULT_RUN_NAME = dt.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_BCE_WITH_LOGITS = True
DEFAULT_LAMBDAS_PIX_LAST = {
    'bce': 30,
    'iou': 0.25,
    'iou_patch': 0.5,
    'mae': 90,
    'mse': 30,
    'triplet': 3,
    'reg': 100,
    'ssim': 10,
    'cnt': 5,
    'structure': 5,
}
DEFAULT_LAMBDAS_PIX_LAST_ACTIVATED = {
    'bce': False,
    'iou': True,
    'iou_patch': False,
    'mae': True,
    'mse': False,
    'triplet': False,
    'reg': False,   
    'ssim': True,
    'cnt': False,
    'structure': False,
}
DEFAULT_LR_DECAY_EPOCHS = [20,40,60,80]
DEFAULT_LR_DECAY_RATE = 0.5
DEFAULT_FINE_TUNE_LAST = 0
GRADIENT_CLIPPING_NORM_DEFAULT = 100.0


class Config():
    def __init__(self, learning_rate=None, bce_with_logits=None, lambdas_pix_last=None, lambdas_pix_last_activated=None, run_name=None, lr_decay_epochs=None, lr_decay_rate=None, fine_tune_last=None) -> None:
        # PATH settings
        # Make up your file system as: SYS_HOME_DIR/codes/dis/BiRefNet, SYS_HOME_DIR/datasets/dis/xx, SYS_HOME_DIR/weights/xx

        #If they are not provided, use the default values
        self.lr = learning_rate if learning_rate is not None else DEFAULT_LR
        self.run_name = run_name if run_name is not None else DEFAULT_RUN_NAME
        self.bce_with_logits = bce_with_logits if bce_with_logits is not None else DEFAULT_BCE_WITH_LOGITS
        self.lambdas_pix_last = lambdas_pix_last if lambdas_pix_last is not None else DEFAULT_LAMBDAS_PIX_LAST
        self.lambdas_pix_last_activated = lambdas_pix_last_activated if lambdas_pix_last_activated is not None else DEFAULT_LAMBDAS_PIX_LAST_ACTIVATED
        self.lr_decay_epochs = lr_decay_epochs if lr_decay_epochs is not None else DEFAULT_LR_DECAY_EPOCHS
        self.lr_decay_rate = lr_decay_rate if lr_decay_rate is not None else DEFAULT_LR_DECAY_RATE
        self.fine_tune_last = fine_tune_last if fine_tune_last is not None else DEFAULT_FINE_TUNE_LAST

        absolute_path = os.path.dirname(__file__)
        self.sys_home_dir = absolute_path.replace('/codes/dis/BiRefNet/birefnet', '')
        self.data_root_dir = os.path.join(self.sys_home_dir, 'datasets/dis')

        # TASK settings
        self.task = 'fine_tuning'

  
        self.prompt4loc = 'dense'

        # Faster-Training settings
        self.load_all = False   # Turn it on/off by your case. It may consume a lot of CPU memory. And for multi-GPU (N), it would cost N times the CPU memory to load the data.
        self.compile = True                             # 1. Trigger CPU memory leak in some extend, which is an inherent problem of PyTorch.
                                                        #   Machines with > 70GB CPU memory can run the whole training on DIS5K with default setting.
                                                        # 2. Higher PyTorch version may fix it: https://github.com/pytorch/pytorch/issues/119607.
                                                        # 3. But compile in Pytorch > 2.0.1 seems to bring no acceleration for training.
        self.precisionHigh = True

        # MODEL settings
        self.gradient_clipping_norm = GRADIENT_CLIPPING_NORM_DEFAULT
        self.ms_supervision = True
        self.out_ref = self.ms_supervision and True
        self.dec_ipt = True
        self.dec_ipt_split = True
        self.cxt_num = 3    # multi-scale skip connections from encoder
        self.mul_scl_ipt = 'cat'
        self.dec_att = 'ASPPDeformable'
        self.squeeze_block = 'BasicDecBlk_x1'
        self.dec_blk = 'BasicDecBlk'
        self.batch_size = 2
        self.start_epoch=245
        self.log_each_steps = 15
        self.lr_warm_up_type = None
        self.display_eval_metrics = ['PA', 'BIoU', 'WF']
        
        self.size = (1024, 1024) # wid, hei
        self.dynamic_size = (0, 0)   # wid, hei. It might cause errors in using compile.
        self.background_color_synthesis = False             # whether to use pure bg color to replace the original backgrounds.
        self.num_workers = max(4, self.batch_size)          # will be decrease to min(it, batch_size) at the initialization of the data_loader

        # Backbone settings
        self.bb = 'swin_v1_l'
        self.lateral_channels_in_collection = [1536, 768, 384, 192]
        if self.mul_scl_ipt == 'cat':
            self.lateral_channels_in_collection = [channel * 2 for channel in self.lateral_channels_in_collection]
        self.cxt = self.lateral_channels_in_collection[1:][::-1][-self.cxt_num:] if self.cxt_num else []

        # MODEL settings - inactive
        self.lat_blk = 'BasicLatBlk'
        self.dec_channels_inter = 'fixed'
        self.refine = ''
        self.progressive_ref = self.refine and True
        self.ender = self.progressive_ref and False
        self.scale = self.progressive_ref and 2
        self.auxiliary_classification = False       # Only for DIS5K, where class labels are saved in `dataset.py`.
        self.refine_iteration = 1
        self.freeze_bb = False
        self.model = 'BiRefNet'

        # TRAINING settings - inactive
        self.preproc_methods = ['flip', 'enhance', 'rotate', 'pepper', 'crop'][:4 if not self.background_color_synthesis else 1]
        self.optimizer = 'AdamW'

        self.lambdas_cls = {
            'ce': 5.0
        }

        # PATH settings - inactive
        self.weights_root_dir = os.path.join(self.sys_home_dir, 'weights/cv')
        self.weights = {
            'pvt_v2_b2': os.path.join(self.weights_root_dir, 'pvt_v2_b2.pth'),
            'pvt_v2_b5': os.path.join(self.weights_root_dir, ['pvt_v2_b5.pth', 'pvt_v2_b5_22k.pth'][0]),
            'swin_v1_b': os.path.join(self.weights_root_dir, ['swin_base_patch4_window12_384_22kto1k.pth', 'swin_base_patch4_window12_384_22k.pth'][0]),
            'swin_v1_l': os.path.join(self.weights_root_dir, ['swin_large_patch4_window12_384_22kto1k.pth', 'swin_large_patch4_window12_384_22k.pth'][0]),
            'swin_v1_t': os.path.join(self.weights_root_dir, ['swin_tiny_patch4_window7_224_22kto1k_finetune.pth'][0]),
            'swin_v1_s': os.path.join(self.weights_root_dir, ['swin_small_patch4_window7_224_22kto1k_finetune.pth'][0]),
            'pvt_v2_b0': os.path.join(self.weights_root_dir, ['pvt_v2_b0.pth'][0]),
            'pvt_v2_b1': os.path.join(self.weights_root_dir, ['pvt_v2_b1.pth'][0]),
        }

        # Callbacks - inactive
        self.verbose_eval = True
        self.only_S_MAE = False
        self.SDPA_enabled = False    # Bugs. Slower and errors occur in multi-GPUs

        # others
        self.device = 0     # .to(0) == .to('cuda:0')

        self.batch_size_test = 1
        self.rand_seed = 7


# Return task for choosing settings in shell scripts.
if __name__ == '__main__':
    config = Config()