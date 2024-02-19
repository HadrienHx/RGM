import numpy as np 

from numpy.linalg import norm
import matplotlib.pyplot as plt

from dataset import Dataset, merge_datasets
from model import LinearRegression, TiltedLinearRegression
from noisers import UnboundedNoiser, ClippingNoiser, NoNoiser
from utils import get_error
import os 
import sys


save_path = ""

# Datasets can be downloaded at: 
# https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/classification.html
data_path = ""


dataset_name = "classification/ijcnn1_full"

path = os.path.join(data_path, dataset_name) 


dataset = Dataset(path, length=1e4)
print(f"Total number of samples: {dataset.n}")
print(f"Dataset dimension: {dataset.d}")

# Parameters 
alpha = 2.          # Rényi-DP alpha
epsilon = 0.1       # Rényi-DP epsilon
n_rounds = 1      # Nb of repetitions (for confidence intervals)
N = 80000000        # Max number of samples
nb_iters = 40       # Number of GD iterations
c = None            # Clipping threshold  
reg = 3e-2          # Regularization
quantile = 1.       # Quantile to set the clipping threshold (rel)
k_filter = 0        # How many times the clip procedure is applied (rel)
plot_clip = True    # Whether to compute and plot clipping baselines

# trunc_datasets, trunc_method = [d.truncate(N) for d in dataset.random_split()], "random"
n_split = 100
trunc_datasets, trunc_method = [d.truncate(N) for d in dataset.targets_split(n_split)], f"targets_{n_split}"

# tilt, is_tilt = np.ones((dataset.d,)) * 1e2, "_bias"
tilt, is_tilt = np.zeros((dataset.d,)), ""

models = [LinearRegression(trunc_datasets[0], reg),
          TiltedLinearRegression(trunc_datasets[1], reg, tilt)]

global_solution = np.linalg.pinv(models[0].A + models[1].A) @ (models[0].b + models[1].b + models[1].A @ tilt)
print(f"Norm of x_star: {np.linalg.norm(global_solution)}")

lr = 0.5 / np.max([model.get_smoothness() for model in models])
print(f"Learning rate: {lr}")

x0 = np.zeros((dataset.d,))

def train(private_algo, models, alpha, epsilon, lr, nb_iters, min_error=0., **kwargs):
    private_models = [
        private_algo(model, alpha, epsilon, **kwargs) 
        for model in models]
    
    errors = []
    for round in range(n_rounds):
        x = np.copy(x0)
        running_error = [get_error(models, x)]
        for i in range(nb_iters):
            x = np.average([
                    private_model.get_new_param(x, lr) 
                    for private_model in private_models
                ], axis=0)
            running_error.append(get_error(models, x))

            if (i+1) % 1000 == 0:
                print(f"Round {round}, iteration {i+1}, error: {running_error[-1]}")
    
        errors.append(running_error)
        if private_algo.name == "NoNoise": 
            break

    errors = np.array(errors) - min_error
    return np.mean(errors, axis=0), np.max(errors, axis=0), np.min(errors, axis=0)


min_error = get_error(models, global_solution)

def rescale_error(error):
    return error - min_error


import seaborn as sns

data = {}

palette = sns.color_palette("colorblind")
x_axis_plots = list(range(nb_iters+1))
alpha_fill = 0.3


mean, e_max, e_min = train(UnboundedNoiser, models, alpha, epsilon, lr, nb_iters, quantile=quantile, filter=0, min_error=min_error)
color = palette[1]
plt.plot(mean, label=f"Relative", color=color)
plt.fill_between(x_axis_plots, e_min, e_max, alpha=alpha_fill, color=color)
data["relative"] = [mean, e_max, e_min]

mean, e_max, e_min = train(NoNoiser, models, alpha, epsilon, lr, nb_iters, min_error=min_error)
color = palette[0]
plt.plot(mean, label=f"No noise", color=color)
plt.fill_between(x_axis_plots, e_min, e_max, alpha=alpha_fill, color=color)
data["nonoise"] = [mean, e_max, e_min]


if plot_clip:
    mean, e_max, e_min = train(ClippingNoiser, models, alpha, epsilon, lr, nb_iters, c=c, c_fac=10., min_error=min_error)
    color = palette[2]
    plt.plot(mean, label=f"Clipped high", color=color) #, linestyle="dashed")
    plt.fill_between(x_axis_plots, e_min, e_max, alpha=alpha_fill, color=color)
    data["largeclip"] = [mean, e_max, e_min]

    mean, e_max, e_min = train(ClippingNoiser, models, alpha, epsilon, lr, nb_iters, c=c, c_fac=1., min_error=min_error)
    color = palette[3]
    plt.plot(mean, label=f"Clipped", color=color) #, linestyle="dashed")
    plt.fill_between(x_axis_plots, e_min, e_max, alpha=alpha_fill, color=color)
    data["clip"] = [mean, e_max, e_min]

    mean, e_max, e_min = train(ClippingNoiser, models, alpha, epsilon, lr, nb_iters, c=c, c_fac=0.1, min_error=min_error)
    color = palette[4]
    plt.plot(mean, label=f"Clipped low", color=color) #, linestyle="dashed")
    plt.fill_between(x_axis_plots, e_min, e_max, alpha=alpha_fill, color=color)
    data["smallclip"] = [mean, e_max, e_min]

plt.semilogy()

plt.legend()
plt.xlabel("Iteration number")
plt.ylabel("f(x) - f*")

eps = str(epsilon/10)[2:]

filename = f"{dataset_name}_{trunc_method}{is_tilt}_{eps}"

if len(sys.argv) > 1:
    filename += "_" + sys.argv[1]

print(filename)

img_path = os.path.join(save_path, "image")
                        
try:
    plt.savefig(os.path.join(img_path, filename))
except: 
    print(f"Could not find directory {img_path}, not saving image")

import pickle

pickle_path = os.path.join(save_path, "data") 
try:
    with open(os.path.join(pickle_path, filename), "wb") as df:
        pickle.dump(data, df)
except: 
    print(f"Could not find directory {pickle_path}, not saving training data")

plt.show()

