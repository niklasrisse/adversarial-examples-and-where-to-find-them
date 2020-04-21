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

def l_1_attack(f_model, x_test, y_test, dataset):

    """
    Carries out an adversarial attack for a given model and dataset that optimizes 
    for closest l_1 distance to the original input. 
    """
         
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

def l_2_attack(f_model, x_test, y_test, dataset):

    """
    Carries out an adversarial attack for a given model and dataset that optimizes 
    for closest l_2 distance to the original input. 
    """
          
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

def l_sup_attack(f_model, x_test, y_test, dataset):

    """
    Carries out an adversarial attack for a given model and dataset that optimizes 
    for closest l_infty distance to the original input. 
    """
  
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

def generate_adversarials(args):

    """
    Generates adversarial examples for a specific dataset (given by args.dataset),
    for different models (given by args.adversarial_model_paths),
    of a specific type (given by args.nn_type),
    with minimal distance to the original data points measured in different norms (given by args.norms)
    """

    NORM_TO_ATTACK = {"inf": l_sup_attack, "2": l_2_attack, "1": l_1_attack}

    _, x_test, _, y_test = dt.get_dataset(args.dataset)

    x_test = x_test[:args.n_points]
    y_test = y_test[:args.n_points]

    n_test_ex, args.height, args.width, args.n_col = x_test.shape
    args.n_in, args.n_out = args.height * args.width * args.n_col, y_test.shape[1]
    args.n_hs = []
    args.seed = 1

    adversarials = dict()

    for adversarial_model_path in args.adversarial_model_paths:

        adversarial_model_name = adversarial_model_path.split("/")[3].split(".mat")[0]

        adversarials[adversarial_model_name] = dict()

        sess = tf.InteractiveSession()

        model, _input, _logits, _activations = load_model(sess, args, adversarial_model_path)

        f_model = foolbox.models.TensorFlowModel(_input,_logits, (0, 1))

        test_predictions = []
        for point in x_test:

            if args.dataset == "mnist" or args.dataset == "fmnist":
                point = point.reshape(1,28,28,1)
            else:
                point = point.reshape(1,32,32,3)

            test_predictions.append(f_model.forward(point).argmax())
                
        test_predictions = np.array(test_predictions)

        for norm in args.norms:

            attack = NORM_TO_ATTACK[norm]
            adversarials[adversarial_model_name][norm] = attack(f_model, x_test, y_test, args.dataset)

        tf.reset_default_graph()
        sess.close()

    return adversarials

def create_noise(adv_pert, norm, noise_type, noise_size):

    """
    Creates a random perturbation of a specific shape (given by adv_pert), a specific type (given by noise_type),
    and a specific size (given by noise_size) measured in a specific norm (given by norm).
    """
        
    if noise_type == "gaussian":
        noise = np.random.normal(size = adv_pert.flatten().shape[0])
    elif noise_type == "uniform":
        noise = np.random.uniform(-1, 1, size = adv_pert.flatten().shape[0])
    else:
        raise Exception("noise type {} not implemented".format(noise_type))

    if norm == np.inf:
        noise = np.sign(noise) * noise_size      
    else:
        noise = (noise / np.linalg.norm(noise, ord=norm)) * noise_size

    return noise

def calculate_perturbation_costs(activations_1, activations_2, norm):

    """
    Calculates the actual perturbation costs for two lists of network activations in a given norm.
    Latex for the perturbation costs: c^i_{(x,y)}(\Delta) := \frac{\|f_i(x + \Delta) - f_i(x)\|}{\|f_i(x)\|}\,.
    """

    perturbation_costs = []
    for (x_i, x_j) in zip(activations_1, activations_2):
        perturbation_costs.append(np.linalg.norm(x_j.flatten() - x_i.flatten(), ord=norm) / np.linalg.norm(x_i.flatten(), ord=norm))

    if (activations_1[-1].argmax() == activations_2[-1].argmax()):
        perturbation_costs.append(0.0)
    else:
        perturbation_costs.append(1.0)
        
    return perturbation_costs

def clean_results(pert_costs_1, pert_costs_2):
        
    """
    Cleans perturbation costs. Removes entrys where the generation of adversarial examples failed.
    """
    
    pert_costs_1 = np.array(pert_costs_1)
    pert_costs_2 = np.array(pert_costs_2)

    n_points_1 = pert_costs_1.shape[0]
    n_points_2 = pert_costs_2.shape[0]

    pert_costs_1 = pert_costs_1[~np.isnan(pert_costs_1).any(axis=1)]
    pert_costs_2 = pert_costs_2[~np.isnan(pert_costs_2).any(axis=1)]

    n_invalid_points_1 = n_points_1 - pert_costs_1.shape[0]
    n_invalid_points_2 = n_points_2 - pert_costs_2.shape[0]

    if (n_invalid_points_1 > n_points_1 * 0.05 or n_invalid_points_2 > n_points_2 * 0.05):
        raise Exception("too many invalid points")

    return pert_costs_1, pert_costs_2

def add_to_pert_cost_data(adversarial_model_name, pert_cost_data, pert_costs_1, pert_costs_2, perturbation_norm, noise_type, noise_size, split):
    
    """
    Adds perturbation costs to the perturbation cost dictionary.
    """

    if adversarial_model_name not in pert_cost_data.keys():
        pert_cost_data[adversarial_model_name] = dict()
    if str(perturbation_norm) not in pert_cost_data[adversarial_model_name].keys():
        pert_cost_data[adversarial_model_name][str(perturbation_norm)] = dict()
    if str(perturbation_norm) not in pert_cost_data[adversarial_model_name][str(perturbation_norm)].keys():
        pert_cost_data[adversarial_model_name][str(perturbation_norm)][str(perturbation_norm)] = dict()
    if noise_type not in pert_cost_data[adversarial_model_name][str(perturbation_norm)][str(perturbation_norm)].keys():
        pert_cost_data[adversarial_model_name][str(perturbation_norm)][str(perturbation_norm)][noise_type] = dict()
    if str(noise_size) not in pert_cost_data[adversarial_model_name][str(perturbation_norm)][str(perturbation_norm)][noise_type].keys():
        pert_cost_data[adversarial_model_name][str(perturbation_norm)][str(perturbation_norm)][noise_type][str(noise_size)] = dict()
    if json.dumps(split) not in pert_cost_data[adversarial_model_name][str(perturbation_norm)][str(perturbation_norm)][noise_type][str(noise_size)].keys():
        pert_cost_data[adversarial_model_name][str(perturbation_norm)][str(perturbation_norm)][noise_type][str(noise_size)][json.dumps(split)] = dict()
    
    pert_cost_data[adversarial_model_name][str(perturbation_norm)][str(perturbation_norm)][noise_type][str(noise_size)][json.dumps(split)]["pert_costs_1"] = pert_costs_1
    pert_cost_data[adversarial_model_name][str(perturbation_norm)][str(perturbation_norm)][noise_type][str(noise_size)][json.dumps(split)]["pert_costs_2"] = pert_costs_2
        

    return pert_cost_data

@time_this
def calculate_perturbation_cost_data(args):

    """
    Calculates the perturbation trajectories for given parameters. 
    Calculates the trajectories for a specific dataset (given by args.dataset),
    for a specific model (given by args.model_path),
    of a specific type (given by args.nn_type),
    with adversarials optimized for different models (given by args.adversarial_model_paths),
    of a specific type (given by args.nn_type),
    with adversarials of minimal distance to the original data points measured in different norms (given by args.norms).
    Optional parameters are the number of points of the dataset to use (given by args.n_points),
    noise types (given by args.noise_types),
    noise sizes (given by args.noise_sizes),
    whether to save the data (given by args.save),
    and whether to plot the data (given by args.plot).
    You can find examples on how to use this method in 'Readme.md' or in the notebooks in the folder 'experiments', 
    which names begin with 'pert_cost_trajectories'.
    """

    print("Starting to calculate perturbation cost data.")

    save_name = "perturbation_cost_trajectories_{}".format(str(datetime.now())[:-7])

    NORM_TO_ORD = {"inf": np.inf, "2": 2, "1": 1}

    adversarials = generate_adversarials(args)

    _, x_test, _, y_test = dt.get_dataset(args.dataset)

    x_test = x_test[:args.n_points]
    y_test = y_test[:args.n_points]

    n_test_ex, args.height, args.width, args.n_col = x_test.shape
    args.n_in, args.n_out = args.height * args.width * args.n_col, y_test.shape[1]
    args.n_hs = []
    args.seed = 1

    sess = tf.InteractiveSession()

    model, _input, _logits, _activations = load_model(sess, args, args.model_path)

    f_model = foolbox.models.TensorFlowModel(_input,_logits, (0, 1))

    perturbation_cost_data = dict()

    for adversarial_model_path in args.adversarial_model_paths:
        adversarial_model_name = adversarial_model_path.split("/")[3].split(".mat")[0]
        for norm in args.norms:
            for noise_type in args.noise_types:
                for noise_size in args.noise_sizes:
                    for split in args.splits:

                        for i in range(x_test.shape[0]):

                            pert_costs_1 = []
                            pert_costs_2 = []

                            perturbation_norm = NORM_TO_ORD[norm]

                            point = x_test[i]

                            if args.dataset == "mnist" or dataset == "fmnist":
                                point = point.reshape(1,28,28,1)
                            else:
                                point = point.reshape(1,32,32,3)

                            activations = sess.run(_activations, feed_dict={_input: point})
                            prediction = np.array([activations[-1].argmax()])

                            adversarial_example = np.array(adversarials[adversarial_model_name][norm][i])
                            
                            if adversarial_example.shape != (1, 28, 28, 1):
                                adversarial_example = np.nan

                            pert = adversarial_example.flatten() - point.flatten()
                            pert_size = np.linalg.norm(pert, ord = perturbation_norm)

                            if noise_size == None:
                                noise = create_noise(pert, perturbation_norm, noise_type, pert_size) 
                                adversarial_activations = sess.run(_activations, feed_dict={_input: adversarial_example})
                            else:
                                noise = create_noise(pert, perturbation_norm, noise_type, noise_size)
                                if adversarial_example.shape == (1, 28, 28, 1):
                                    adversarial_example = (adversarial_example.flatten() + noise).reshape(adversarial_example.shape)
                                adversarial_activations = sess.run(_activations, feed_dict={_input: adversarial_example})
                            
                            point_plus_noise = (point.flatten() + noise).reshape(point.shape)

                            noisy_activations = sess.run(_activations, feed_dict={_input: point_plus_noise})
                            
                            if (pert_size > split[norm][0] and pert_size < split[norm][1]):
                                
                                perturbation_costs = calculate_perturbation_costs(activations, adversarial_activations, perturbation_norm)
                                pert_costs_1.append(perturbation_costs.copy())
                            
                                perturbation_costs = calculate_perturbation_costs(activations, noisy_activations, perturbation_norm)
                                pert_costs_2.append(perturbation_costs.copy())
                            
                        
                        pert_costs_1, pert_costs_2 = clean_results(pert_costs_1, pert_costs_2)
                        perturbation_cost_data = add_to_pert_cost_data(adversarial_model_name, perturbation_cost_data, pert_costs_1, pert_costs_2, norm, noise_type, noise_size, split)

    tf.reset_default_graph()
    sess.close()

    if args.save == True:
        save_to_json(perturbation_cost_data, save_name)


    return perturbation_cost_data

def plot_perturbation_cost_data(perturbation_cost_data):

    """
    Plots the perturbation cost data.
    """

    save_name = "perturbation_cost_trajectories_{}".format(str(datetime.now())[:-7])

    number_of_trajectories = 0
    for key, value in perturbation_cost_data.items():
        for key2, value2 in value.items():
            for key3, value3 in value2.items():
                for key4, value4 in value3.items():
                    for key5, value5 in value4.items():
                        for key6, value6 in value5.items():
                            number_of_trajectories += 2
            

    fig, ax = plt.subplots(1, 1, figsize = (6, 5), dpi=400)
    colors = sns.color_palette(n_colors=number_of_trajectories)

    norm_to_latex = {"inf":"\infty", "2":"2", "1": "1"}

    k = 0
    for adversarial_model_name, value in perturbation_cost_data.items():
        for perturbation_norm, value2 in value.items():
            for perturbation_cost_norm, value3 in value2.items():
                for noise_type, value4 in value3.items():
                    for noise_size, value5 in value4.items():
                        for split, value6 in value5.items():
                            ax.set_xticks(np.arange(0, value6["pert_costs_1"][0].shape[0], 1.0))
                            ax.plot(np.mean(value6["pert_costs_1"], axis=0), c = colors[k], label = "adv. noise, norm=$\ell_{}$".format(norm_to_latex[perturbation_norm]))
                            k += 1
                            if noise_size=="None":
                                noise_size="same"
                            ax.plot(np.mean(value6["pert_costs_2"], axis=0), c = colors[k], label = "random noise, norm=$\ell_{}$, type={}, size={}".format(norm_to_latex[perturbation_norm], noise_type, noise_size))                    
                            k += 1

    ax.legend()
    ax.set_ylabel("layer")
    ax.set_xlabel("perturbation costs")
    ax.set_title("perturbation cost trajectories")

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
    parser.add_argument('--adversarial_model_paths', nargs='+' , help='Model paths, for which the adversarial examples shall be generated.', required=False)
    parser.add_argument('--norms', nargs='+', help='Give multiple norms. Currently supported: {2, 1, inf}.', required=True)
    parser.add_argument('--noise_types', nargs='+', default=["gaussian"], help='Give multiple noise types. Currently supported: {gaussian, uniform}.')
    parser.add_argument('--noise_sizes', nargs='+', default=[None], help='Give multiple noise sizes.')
    parser.add_argument('--save', type=bool , default='True', help='Specify whether to save the robustness curve data.')
    parser.add_argument('--plot', type=bool , default='True', help='Specify whether to plot the robustness curve data.')

    args = parser.parse_args()
    args.splits = [{"inf": [0.0, np.inf], "2": [0.0, np.inf]}]
    if args.adversarial_model_paths == None:
        args.adversarial_model_paths = [args.model_path]

    perturbation_cost_data = calculate_perturbation_cost_data(args)

    if args.plot == True:
        plot_perturbation_cost_data(perturbation_cost_data)

