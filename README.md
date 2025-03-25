Forked BiRefNet repository

Added train_finetuning.sh file, which contains the correct settings of the file train.sh, to correctly running the training.

The checkpoint are saved starting from the last save_last epochs and each save_step epochs.

Initially we had an out of memory problem when saving the checkpoint in the last epochs. We solved it, using the cpu instead of the gpu to save the checkpoint.