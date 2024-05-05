'''
Iterates through ranges of hyperparameters in config.py and runs the model using training.py.
'''
import os
import itertools
from time import time
from datetime import datetime
import argparse
import csv
import sys

import numpy as np
import jax
import pennylane as qml
import optax
import matplotlib.pyplot as plt

from config import get_args
import architecture as arch
import load
from training import gradient_and_cost, circuit, send_email, handle_exception

# sys.excepthook = handle_exception

# rewrite stop.txt to "no" to continue running
with open("stop.txt", "w", encoding="utf-8") as f:
    f.write("no")

# Get the arguments from the command line or from the default values in config.py, as well as the parser
args, parser = get_args()

# Configure JAX
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_default_device', jax.devices('gpu')[args.device])

# Get the default values of the arguments
defaults = parser.parse_args([])

# Make a dictionary of args that are not default
nondefaults = {}
for arg in vars(args):
    if getattr(args, arg) != getattr(defaults, arg):
        nondefaults[arg] = getattr(args, arg)

# set list of arguments that circuit and vectorization do not depend on - only classical
classical_args = ["rates", "iterations"]

# Separate the static and changing arguments
static_args = {}
changing_args = {}
for arg in vars(args):
    value = getattr(args, arg)
    if not isinstance(value, list) or (isinstance(value, list) and len(value) == 1):
        static_args[arg] = value if not isinstance(value, list) else value[0]
    else:
        changing_args[arg] = value

# Create folder with names all the non-default arguments
# add a Run_id with date and time stamp to avoid overwriting
# example: results/_filters_1-2_convolutiontypes_False-True_pooltypes_trace-separate_2021-08-25_15-30-00
if args.testing is False:
    t = datetime.now().strftime("%Y-%m-%d_%H-%M")
    folder_name = f"results/{t}_"
    for arg, values in nondefaults.items():
        if isinstance(values, list):
            if len(values) == 1:
                folder_name += arg + "_" + str(values[0]) + "_"
            else:
                folder_name += arg + "_" + str(values[0]) + "-" + str(values[-1]) + "_"
        else:
            folder_name += arg + "_" + str(values) + "_"
    os.mkdir(folder_name)

# set jax and numpy random seeds, save it to the results.txt file
seed = int(time())
key = jax.random.PRNGKey(seed)
np.random.seed(seed)
if args.testing is False:
    with open(folder_name + "/results.txt", "a", encoding="utf-8") as f:
        f.write("Random seed: " + str(seed) + "\n")

    # Save the arguments in a text file summarizing the run
    with open(folder_name + "/results.txt", "a", encoding="utf-8") as f:
        f.write("Run with arguments:\n")
        for arg, value in static_args.items():
            f.write(arg + ": " + str(value) + "\n")
        for arg, value in changing_args.items():
            f.write(arg + ": " + str(value) + "\n")

# Get all combinations of changing arguments
combinations = list(itertools.product(*changing_args.values()))

if args.testing is False:
    # Track acccuracy for each combination in a csv file
    with open(folder_name + "/accuracies.csv", "w", encoding="utf-8") as f:
        f.write("Accuracy,")
        for arg in changing_args:
            f.write(arg + ",")
        f.write("1st run time,Later run times")

prev_args = None
prev_image_shape = None

# Run your script for each combination
for combination in combinations:

    # track the time
    times = [time()]

    # Create a new args object with the static arguments and one combination of the changing arguments
    new_args = argparse.Namespace(**static_args, **dict(zip(changing_args.keys(), combination)))

    if args.testing is False:
        # create a subfolder for each combination
        subfolder_name = folder_name + "/"
        if len(changing_args) > 0:
            for arg in changing_args:
                subfolder_name += arg + "_" + str(getattr(new_args, arg)) + "_"
            subfolder_name = subfolder_name[:-1]
            os.mkdir(subfolder_name)

    # Load the data
    train_data = load.LoadData(new_args.datasets,
                               train=True,
                               filters=new_args.filters,
                               classes=new_args.classes,
                               batch_size=new_args.batchsizes,
                               iters=new_args.iterations)
    test_data = load.LoadData(new_args.datasets,
                              train=False,
                              filters=new_args.filters,
                              classes=new_args.classes,
                              batch_size=new_args.testbatch)
    
    if prev_args is not None:
        quantum_or_vectorization_args_changed = any(getattr(new_args, arg) != getattr(prev_args, arg) for arg in changing_args if arg not in classical_args)

        image_shape_changed = train_data.image_shape != prev_image_shape

    # Skip redefining the architecture parameters if only learning rates or iterations are changing, or if the dataset has not changed in size
    if prev_args is None or quantum_or_vectorization_args_changed or image_shape_changed:
        # Define the architecture parameters
        conv_params = arch.ConvParams(new_args.filters, new_args.convolutiontypes)
        pool_params = arch.PoolParams(new_args.pooltypes, new_args.separate, new_args.doublepool)
        if new_args.imprimitives is not None:
            for idx, imprimitive in enumerate(new_args.imprimitives):
                try:
                    new_args.imprimitives[idx] = getattr(qml, imprimitive)
                except TypeError:
                    pass
        dense_params = arch.DenseParams(new_args.structurestring, int(np.ceil(np.log2(train_data.classes))), new_args.ranges, new_args.imprimitives, new_args.dense_pool_type)

        # setup the quantum device and load the circuit on it. Also define vectorized and jitted versions of the circuit
        wires = int(np.log2(np.prod(np.array(train_data.image_shape))))
        dev = qml.device('default.qubit', wires=wires)
        qnode = qml.QNode(circuit, dev, interface='jax-jit')
        vmap_qnode = jax.vmap(qnode, in_axes=(0, None, None, None, None, None))
        vmap_qnode_jit = jax.jit(vmap_qnode, static_argnums=(1, 3, 4, 5))
        vmap_jacobian = jax.vmap(jax.jacobian(qnode, argnums=2), in_axes=(0, None, None, None, None, None))

    # loop the required number of training runs and initialize the parameters separately for each run
    nparams = arch.QCNN.nparams(list(range(wires)), conv_params=conv_params, pool_params=pool_params, dense_params=dense_params)

    if args.testing is False:
        # save the arguments in a text file
        with open(subfolder_name + "/results.txt", "a", encoding="utf-8") as f:
            f.write(f"{nparams}-parameter model for {train_data.classes}-class classification of {new_args.datasets} dataset\n")
            f.write(f"Iterations - {train_data.datapoints//new_args.batchsizes if new_args.iterations is None else new_args.iterations}\t"
                    f"Learning rate - {new_args.rates}\tBatch size - {new_args.batchsizes}\tClasses - {train_data.classes if new_args.classes is None else new_args.classes}\n")
            f.write(f"Train size - {train_data.datapoints}\tTest size - {test_data.datapoints}\t{new_args.trainingruns}-trial average\n")
            f.write("Random seed: " + str(seed) + "\n")
            f.write(f"Filters - {new_args.filters}\tConvolutional parameters - {82 * (3**new_args.convolutiontypes)}\n")
            f.write(f"Pooling - {new_args.pooltypes}\tSeparate pooling parameters - {new_args.separate}\tDouble pooling - {new_args.doublepool}\n")
            f.write(f"Structure string - {new_args.structurestring}\tRanges - {new_args.ranges}\tImprimitives - {new_args.imprimitives}\tDense pooling type - {new_args.dense_pool_type}\n")
    
    # track accuracy for each trial
    trial_accuracies = []

    # loop over the trials
    trial = 0
    while trial < new_args.trainingruns:
        trial += 1

        # initialize the parameters
        key, subkey = jax.random.split(key)
        params = 2 * np.pi * jax.random.uniform(key, [nparams]).astype(float)

        if args.testing is False:
            # make a subfolder for current trial
            trial_folder_name = "/Trial_" + str(trial)
            os.mkdir(subfolder_name + trial_folder_name)

            # save the parameters and run conditions in a text file
            with open(subfolder_name + trial_folder_name + "/results.txt", "a", encoding="utf-8") as f:
                f.write(f"{nparams}-parameter model for {train_data.classes}-class classification of {new_args.datasets} dataset\n")
                f.write(f"Iterations - {new_args.iterations}\tLearning rate - {new_args.rates}\tBatch size - {new_args.batchsizes}\t")
                f.write(f"Train size - {train_data.datapoints}\tTest size - {test_data.datapoints}\tTrial - {trial}\n")
                f.write("Random seed: " + str(seed) + "\n")
                f.write(f"Filters - {new_args.filters}\tConvolutional parameters - {82 * (3**new_args.convolutiontypes)}\n")
                f.write(f"Pooling - {new_args.pooltypes}\tSeparate pooling parameters - {new_args.separate}\tDouble pooling - {new_args.doublepool}\n")
                f.write(f"Structure string - {new_args.structurestring}\tRanges - {new_args.ranges}\tImprimitives - {new_args.imprimitives}\tDense pooling type - {new_args.dense_pool_type}\n")
                f.write("Start parameters:\n")
                f.write(" ".join(map(str, params))+ "\n")

        # initialize the optimizer and cost array
        optimizer = optax.adam(new_args.rates)
        opt_state = optimizer.init(params)
        costs = []

        epoch = 0
        # loop over the epochs
        for images, labels in train_data:
            with open("stop.txt", "r", encoding="utf-8") as f:
                if f.read() == "now" or f.read() == "next":
                    break
            if epoch % 200 == 0:
                print(f"Trial {trial}, Epoch {epoch} at {time()-seed} seconds.")
            # evaluate the gradient and cost function using the quantum device
            gradients, cost = gradient_and_cost(vmap_qnode, vmap_jacobian, images, labels, params, conv_params, pool_params, dense_params)
            # update the parameters
            updates, opt_state = optimizer.update(gradients, opt_state)
            params = optax.apply_updates(params, updates)
            costs.append(cost)
            epoch += 1
        
        # end the trial is args.testing is True
        if args.testing is True:
            break

        # save the parameters in a text file
        with open(subfolder_name + trial_folder_name + "/results.txt", "a", encoding="utf-8") as f:
            f.write("Optimized parameters:\n")
            f.write(" ".join(map(str, params))+ "\n")

        # plot the cost function and save it
        plt.plot(costs)
        plt.xlabel("Iterations")
        plt.ylabel("Cross-entropy loss")
        plt.title("Cost function")
        plt.savefig(subfolder_name + trial_folder_name + "/cost.png")
        plt.clf()
        plt.cla()
        plt.close()

        # evaluate the accuracy and other metrics of the model on the test data
        count = 0
        for images, labels in test_data:
            count += 1
            if count % 100 == 0:
                print(f"Trial {trial}, Test batch {count} at {time()-seed} seconds.")
            if count == 1:
                probs = vmap_qnode_jit(images, range(wires), params, conv_params, pool_params, dense_params)
                lbls = labels
                continue
            probs = np.vstack((probs, vmap_qnode_jit(images, range(wires), params, conv_params, pool_params, dense_params)) )
            lbls = np.append(lbls, labels)

        predictions = np.argmax(probs, axis=1)

        accuracy = np.mean(predictions == lbls)
        class_accuracies = [np.mean(predictions[lbls == i] == i) for i in range(train_data.classes)]

        prob_correct = np.max(probs, axis=1)

        confusion = np.array([
                            [np.sum((predictions == i) & (lbls == j)) for j in range(train_data.classes)] for i in range(train_data.classes)
                            ])

        # save the accuracies in a text file
        with open(subfolder_name + trial_folder_name + "/results.txt", "a", encoding="utf-8") as f:
            f.write(f"Accuracy - {accuracy}\tClasswise accuracy - {class_accuracies}\n")
            f.write(f"Expected correct probability - {np.mean(prob_correct)}+-{np.std(prob_correct)/np.sqrt(test_data.datapoints)}\tMax - {np.max(prob_correct)}\tMin - {np.min(prob_correct)}\n")
            f.write(f"Confusion matrix - \n{confusion}\n")

        if new_args.trainperf:
            # evaluate the accuracy and other metrics of the model on the training data
            count = 0
            for images, labels in train_data:
                count += 1
                print(f"Trial {trial}, Train batch {count} at {time()-seed} seconds.")
                if count == 1:
                    train_probs = vmap_qnode_jit(images, range(wires), params, conv_params, pool_params, dense_params)
                    train_lbls = labels
                    continue
                train_probs = np.vstack((train_probs, vmap_qnode_jit(images, range(wires), params, conv_params, pool_params, dense_params)) )
                train_lbls = np.append(train_lbls, labels)

            train_predictions = np.argmax(train_probs, axis=1)

            train_accuracy = np.mean(train_predictions == train_lbls)
            train_class_accuracies = [np.mean(train_predictions[train_lbls == i] == i) for i in range(train_data.classes)]

            train_prob_correct = np.max(train_probs, axis=1)

            train_confusion = np.array([
                [
                    np.sum((train_predictions == i) & (train_lbls == j)) 
                    for j in range(train_data.classes)
                ] 
                for i in range(train_data.classes)
            ])            
            # save the accuracies in a text file

            with open(subfolder_name + trial_folder_name + "/results.txt", "a", encoding="utf-8") as f:
                f.write(f"Train accuracy - {train_accuracy}\t"
                        f"Train classwise accuracy - {train_class_accuracies}\n")
                f.write(f"Train expected correct probability -"
                        f"{np.mean(train_prob_correct)}+-"
                        f"{np.std(train_prob_correct)/np.sqrt(train_data.datapoints)}\t"
                        f"Max - {np.max(train_prob_correct)}\tMin - {np.min(train_prob_correct)}\n")
                f.write(f"Train confusion matrix - \n{train_confusion}\n")

        # save the time taken for the trial to the list, and save it in the results.txt file
                
        times.append(time())
        with open(subfolder_name + trial_folder_name + "/results.txt", "a", encoding="utf-8") as f:
            f.write(f"Time taken - {times[-1]-times[-2]} seconds\n")

        # save the accuracies to the trackers
        trial_accuracies.append(accuracy)

        with open("stop.txt", "r", encoding="utf-8") as f:
            if f.read() in ["now", "trial", "next"]:
                break


    # evaluate average accuracy over the trials
    average_accuracy = np.mean(trial_accuracies)

    if args.testing is False:
        # save accuracies and average accuracy in results.txt
        with open(subfolder_name + "/results.txt", "a", encoding="utf-8") as f:
            f.write(f"Average accuracy - {average_accuracy}+-{np.std(trial_accuracies)/np.sqrt(new_args.trainingruns)}\n")
            f.write(f"Trial accuracies - {trial_accuracies}\n")

        # save accuracies and time in csv file
        with open(folder_name + "/accuracies.csv", "a", encoding="utf-8") as f:
            f.write(f"\n{average_accuracy},")
            for arg in changing_args:
                if arg in ["ranges", "imprimitives"]:
                    f.write(f"\"{getattr(new_args, arg)}\",")
                else:
                    f.write(f"{getattr(new_args, arg)},")
            if trial > 1:
                f.write(f"{times[-trial]-times[-trial-1]},{np.mean(np.diff(times[1-trial:]))}")

    prev_args = new_args
    prev_image_shape = train_data.image_shape

    with open("stop.txt", "r", encoding="utf-8") as f:
        if f.read() in ["now", "trial", "combination"]:
            break

if args.testing is False:
    # sort the accuracies.csv file by the first column
    # Read the header and data from the file
    with open(folder_name + "/accuracies.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # Save the header
        data = list(reader)

# Define the data type for each column
dtypes = [('col1', float)] + [('col'+str(i), 'O') for i in range(2, len(data[0])+1)]

# Convert the data to a structured numpy array
data = np.array([tuple(row) for row in data], dtype=dtypes)

# Sort the data by the first column
data = np.sort(data, order='col1')

if args.testing is False:
    # Write the header and sorted data back to the file
    with open(folder_name + "/accuracies.csv", "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)  # Write the header
        for row in data:
            writer.writerow(row)  # Write the sorted data

# send an email to the user when the script is done

subject = "Quantum CNN training complete"
message = f"Best accuracy - {data[-1][0]}\n"
filename = folder_name + "/accuracies.csv"

send_email(subject, message, filename)
