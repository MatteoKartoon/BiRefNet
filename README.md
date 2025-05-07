Forked BiRefNet repository

Added train_finetuning.sh file, which contains the correct settings of the file train.sh, to correctly running the training.
Added evaluations.py file, which contains the correct settings of the file eval_existingOnes.py, to correctly run the evaluation.

Separate the code into three folders:
-scripts for the python scripts
-bash_scripts for the bash scripts
-birefnet for the libraries file

Implemented code:
- BiRefNet repository
    - 5 fundamental python scripts (each called by a corresponding bash script)
        - Splitting: take two folders (original and annotated images) —>split them into train, validation and test
        - Training: train neural network —>save checkpoints in a folder
        - Testing: take a checkpoint—>save predictions in a folder
        - Evaluation: take some model predictions —> compute the metrics on the testset
        - Visualization: take model predictions (fine-tuned model/current model/photoroom)  and choose some metrics —> visualize the predictions and ranking the models with respect to the chosen metrics
    - 3 python scripts to compute:
        - Baseline model predictions, given an RGB folder
        - The alpha channel masks, given an RGBA annotated images folder
        - Photoroom predictions, given an RGB folder

To get a working training I switched the data type to bfloa16, otherwise I had a nan gradient problem.