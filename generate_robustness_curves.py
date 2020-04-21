import os
import time
import json
import argparse
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import foolbox
import scipy.io as io
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from utils import L1, L2, NumpyEncoder
from provable_robustness_max_linear_regions import data as dt
from provable_robustness_max_linear_regions import models
from provable_robustness_max_linear_regions.models import load_model

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('text', usetex=True)



def time_this(original_function):
    
    """
    Wraps a timing function around a given function.
    """

    def new_function(*args, **kwargs):
        
        timestart = time.time()                  
        x = original_function(*args, **kwargs)               
        timeend = time.time()                     
        print("Took {0:.2f} seconds to run.\n".format(timeend-timestart))
            
        return x           
        
    return new_function

def save_to_json(dictionary, file_name):

    """
    Saves a given dictionary to a json file.
    """
        
    if not os.path.exists("res"):
        os.makedirs("res")

    with open("res/" + file_name + ".json", 'w') as fp:
        json.dump(dictionary, fp, cls = NumpyEncoder)

@time_this
def l_1_attack(f_model, x_test, y_test, dataset):

    """
    Carries out an adversarial attack for a given model and dataset that optimizes 
    for closest l_1 distance to the original input. 
    """
         
    print("Starting l_1 attack.")
    attack = foolbox.attacks.EADAttack(model = f_model, criterion = foolbox.criteria.Misclassification(), distance = L1)
    
    adversarials = []
    for i, point in enumerate(x_test):

        if dataset == "mnist" or dataset == "fmnist":
            point = point.reshape(1,28,28,1)
        else:
            point = point.reshape(1,32,32,3)

        adversarials.append(attack(point, np.array([y_test[i].argmax()]), binary_search_steps=10))
        
    adversarials = np.array(adversarials)
        
    return adversarials

@time_this
def l_2_attack(f_model, x_test, y_test, dataset):

    """
    Carries out an adversarial attack for a given model and dataset that optimizes 
    for closest l_2 distance to the original input. 
    """
          
    print("Starting l_2 attack.")
    attack = foolbox.attacks.CarliniWagnerL2Attack(model = f_model, criterion = foolbox.criteria.Misclassification(), distance = L2)
    
    adversarials = []
    for i, point in enumerate(x_test):

        if dataset == "mnist" or dataset == "fmnist":
            point = point.reshape(1,28,28,1)
        else:
            point = point.reshape(1,32,32,3)

        adversarials.append(attack(point, np.array([y_test[i].argmax()]), binary_search_steps=10))
        
    adversarials = np.array(adversarials)
        
    return adversarials

@time_this
def l_sup_attack(f_model, x_test, y_test, dataset):

    """
    Carries out an adversarial attack for a given model and dataset that optimizes 
    for closest l_infty distance to the original input. 
    """

    print("Starting l_sup attack.")    
    attack = foolbox.attacks.ProjectedGradientDescentAttack(model = f_model, criterion = foolbox.criteria.Misclassification(), distance = foolbox.distances.Linf)
    
    adversarials = []
    for i, point in enumerate(x_test):

        if dataset == "mnist" or dataset == "fmnist":
            point = point.reshape(1,28,28,1)
        else:
            point = point.reshape(1,32,32,3)

        adversarials.append(attack(point, np.array([y_test[i].argmax()])))
        
    adversarials = np.array(adversarials)
        
    return adversarials

def generate_curve_data(args):

    """
    Calculates the robustness curve data for given parameters. 
    Calculates the data for a specific dataset (given by args.dataset),
    for a specific model (given by args.model_path),
    of a specific type (given by args.nn_type),
    with adversarials of minimal distance to the original data points measured in different norms (given by args.norms).
    Optional parameters are the number of points of the dataset to use (given by args.n_points),
    whether to save the data (given by args.save),
    and whether to plot the data (given by args.plot).
    You can find examples on how to use this method in 'Readme.md' or in the notebooks in the folder 'experiments', 
    which names begin with 'rob_curves'.
    """

    save_name = "approximated_robustness_curves_{}".format(str(datetime.now())[:-7])

    NORM_TO_ATTACK = {"inf": l_sup_attack, "2": l_2_attack, "1": l_1_attack}

    _, x_test, _, y_test = dt.get_dataset(args.dataset)

    x_test = x_test[:args.n_points]
    y_test = y_test[:args.n_points]

    n_test_ex, args.height, args.width, args.n_col = x_test.shape
    args.n_in, args.n_out = args.height * args.width * args.n_col, y_test.shape[1]
    
    if args.nn_type == "cnn":
        args.n_hs = []
    else:
        args.n_hs = [1024]
    args.seed = 1

    sess = tf.InteractiveSession()

    model, _input, _logits, _ = load_model(sess, args, args.model_path)

    f_model = foolbox.models.TensorFlowModel(_input,_logits, (0, 1))

    test_predictions = []
    for point in x_test:

        if args.dataset == "mnist" or args.dataset == "fmnist":
            point = point.reshape(1,28,28,1)
        else:
            point = point.reshape(1,32,32,3)

        test_predictions.append(f_model.forward(point).argmax())
            
    test_predictions = np.array(test_predictions)

    robustness_curve_data = dict()

    for norm in args.norms:

        attack = NORM_TO_ATTACK[norm]
        adversarials = attack(f_model, x_test, y_test, args.dataset)

        dists_r = np.array([np.linalg.norm(x = vector, ord = np.inf) for vector in np.subtract(adversarials.reshape(adversarials.shape[0],-1 ), x_test.reshape(x_test.shape[0], -1))])
        dists_r[test_predictions != y_test.argmax(axis=1)] = 0
        dists_r.sort(axis=0)

        probs = 1/float(test_predictions.shape[0]) * np.arange(1, test_predictions.shape[0]+1)

        probs[np.isnan(dists_r)] = 1.0
        dists_r = np.nan_to_num(dists_r, nan = np.nanmax(dists_r))

        robustness_curve_data[norm] = {"x" : dists_r, "y": probs }

    if args.save == True:
        save_to_json(robustness_curve_data, save_name)

    tf.reset_default_graph()
    sess.close()

    return robustness_curve_data

def plot_curve_data(robustness_curve_data):

    """
    Plots the robustness curve data.
    """

    save_name = "approximated_robustness_curves_{}".format(str(datetime.now())[:-7])

    fig, ax = plt.subplots(1, 1, figsize = (6, 5), dpi=400)

    norms = robustness_curve_data.keys()
    colors = sns.color_palette(n_colors=len(norms))
    norm_to_latex = {"inf":"\infty", "2":"2", "1": "1"}

    for i, norm in enumerate(norms):

        robustness_curve_data[norm]["x"] = np.insert(robustness_curve_data[norm]["x"], 0, 0.0, axis=0)
        robustness_curve_data[norm]["y"] = np.insert(robustness_curve_data[norm]["y"], 0, 0.0, axis=0)

        ax.plot(robustness_curve_data[norm]["x"], robustness_curve_data[norm]["y"] * 100, c = colors[i], label = "$\ell_{}$ robustness curve".format(norm_to_latex[norm]))

    ax.legend()
    ax.set_ylabel("test set loss in $\%$")
    ax.set_xlabel("perturbation size")
    ax.set_title("robustness curves")
    ax.set_xlim(left=0.0)

    fig.tight_layout()

    if not os.path.exists("res"):
        os.makedirs("res")

    fig.savefig('res/{}.pdf'.format(save_name))

    plt.show()

if __name__ == "__main__":
    np.random.seed(1)

    parser = argparse.ArgumentParser(description='Define parameters.')
    parser.add_argument('--dataset', type=str, help='Dataset. Currently supported: {mnist, fmnist, gts, cifar10}.', required=True)
    parser.add_argument('--n_points', type=int, default=-1, help='Number of points of dataset to use.')
    parser.add_argument('--model_path', type=str , help='Model path.', required=True)
    parser.add_argument('--nn_type', type=str , help='Type of neural network. Currently supported: {cnn, fc1}.', required=True)
    parser.add_argument('--norms', nargs='+', help='Give multiple norms. Currently supported: {2, 1, inf}.', required=True)
    parser.add_argument('--save', type=bool , default='True', help='Specify whether to save the robustness curve data.')
    parser.add_argument('--plot', type=bool , default='True', help='Specify whether to plot the robustness curve data.')


    args = parser.parse_args()

    robustness_curve_data = generate_curve_data(args)

    if args.plot == True:
        plot_curve_data(robustness_curve_data)

