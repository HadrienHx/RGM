## Code to run experiments

This repository contains the code for the AISTATS 2024 submission "The Relative Gaussian Mechanism
and its Application to Private Gradient Descent". Details and theory can be found in the paper.  

# Requirements

Create a new python 3 environment:

`conda create --name rgm_env python=3.9`

Then, switch to this environment using:

`conda activate rgm_env`

Install the following packages:

`conda install matplotlib seaborn numba scikit-learn numpy scipy`


# Run the code

To run the code, and plot the results, simply use the command:

`python main.py`

Extra arguments can be passed to name the output files if necessary. 

# Configuration

Configuration can be changed directly in the `main.py` file. 

In particular, it is necessary to change the `data_path` and `save_path` options: 

- Change `data_path` option such that `data_path/dataset_name` exists, and corresponds to the dataset you would like to process.

- Change `save_path` such that the directories `save_path/data` and `save_path/image` exist, to store output files from training (for easy plotting), and the output pdf image.
    
