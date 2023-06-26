This is the repo for intracardiac dataset task 3.

## Repo Structure
+ config: training config files

+ data: your dataset folder, maybe use a symbolic link. (*e.g.* I use `data/intracardiac_dataset`)

+ file: experiment files path (*e.g.* where to store the training log and checkpoints)

+ src: source code.
    + dataloader: module to load the intracardiac dataset.
    + experiment: executable files.
    + model：module of DIY-ed squeezenet.
    + tool: some helpful tools.

## Requirements
```
conda install pytorch=1.12.1 torchvision pytorch-cuda=11.4 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Example Usage
```
python src/experiment/task3_training.py --config-file config/demo.json
```
