# Task 4 Directions 

Here's a general overview of what's in the task4 directory.

## intracardiac_dataset 

This is a folder that needs to contain all of the data we'll be using. Currently empty, so be sure to either run task_4_getting_started.ipynb or move the files into this directory yourself.

## download_intracardiac_dataset.sh 

Used in task_4_getting_started to download the data. 

## code 

The main directory you'll be concerned with. 


### task_4_getting_started.ipynb

This is for getting setup. If you're working on a remote server, use this to download the intracardiac dataset. NOTE: other code will fail if intracardiac_dataset is not in the correct location. Either move intracardiac_dataset to the task4 directory, or run this notebook.

### cardiac_ml_tools.py 

Contains some helpful tools. 

### t4_paper 

Code that tries to duplicate the model from Mikel's paper. There's a .py version and a jupyter notebook version. Run the python script if you only care about the end result, go through the jupyter notebook if you want more commentary. 

For a more in depth review of what specific code does, I recommend perusing the jupyter notebooks. 