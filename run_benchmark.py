from glob import glob
from tqdm import tqdm
from datetime import datetime
from collections import deque
import pathlib
import os
import torch
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
import pandas as pd

from gravann import directional_encoding, positional_encoding, direct_encoding, spherical_coordinates
from gravann import normalized_loss, mse_loss
from gravann import ACC_ld, U_mc, U_ld, U_trap_opt, sobol_points
from gravann import U_L
from gravann import enableCUDA, max_min_distance
from gravann import get_target_point_sampler
from gravann import init_network, train_on_batch
from gravann import create_mesh_from_cloud, plot_model_vs_cloud_mesh, plot_model_rejection

EXPERIMENT_ID = "run_27_10_2020"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"            # Select GPUs
OUTPUT_FOLDER = "results/" + EXPERIMENT_ID + "/"    # Results folder
SAMPLE_PATH = "mascons/"                            # Mascon folder
# Number of training iterations
ITERATIONS = 3000
# SAMPLES = glob(SAMPLE_PATH + "/*.pk")             # Use all available samples
SAMPLES = [                                         # Use some specific samples
    "mascons/Eros.pk",
    #"mascons/Churyumov   ^`^sGerasimenko.pk",
    # "mascons/Itokawa.pk",
    # "mascons/sample_01_cluster_2400.pk",
    # "mascons/sample_04_cluster_6674_hollow_0.3_0.3.pk",
    # "mascons/sample_08_cluster_1970.pk"
]

N_INTEGR_POINTS = 800000                # Number of integrations points for U
TARGET_SAMPLER = [  # "spherical",          # How to sample target points
    "cubical",
]
SAMPLE_DOMAIN = [1.0,                   # Defines the distance of target points
                 1.1]
BATCH_SIZES = [1000]                     # For training
LRs = [1e-4]                            # LRs to use
LOSSES = [                              # Losses to use
    normalized_loss,
    mse_loss
]

ENCODINGS = [                           # Encodings to test (positional currently N/A because it needs one more parameter)
    directional_encoding,
    direct_encoding,
    # spherical_coordinates
]
USE_ACC = False                         # Use acceleration instead of U (TODO)
INTEGRATOR = U_trap_opt
ACTIVATION = [                          # Activation function on the last layer
    torch.nn.Sigmoid(),
    # torch.nn.Softplus(),
    # torch.nn.Tanh(),
    # torch.nn.LeakyReLU(),
]
SAVE_PLOTS = True                       # If plots should be saved.

RESULTS = pd.DataFrame(columns=["Sample", "Type", "Loss", "Encoding", "Integrator", "Activation",
                                "Batch Size", "LR", "Target Sampler", "Integration Points", "Final Loss",
                                "Final Running Loss", "Final WeightedAvg Loss"])


TOTAL_RUNS = len(ACTIVATION) * len(ENCODINGS) * len(LOSSES) * \
    len(LRs) * len(BATCH_SIZES) * len(TARGET_SAMPLER) * len(SAMPLES)


def run():
    """This function runs all the permutations of above settings
    """
    print("#############  Initializing    ################")
    print("Using the following samples:", SAMPLES)
    print("###############################################")
    enableCUDA()
    print("Will use device ", os.environ["TORCH_DEVICE"])
    print("###############################################")
    # Make output folders
    print("Making folder structre...", end="")
    _make_folders()
    print("###############################################")

    run_counter = 0  # Counting number of total runs

    for sample in SAMPLES:
        print(f"\n--------------- STARTING {sample} ----------------")
        points, masses = _load_sample(sample)

        # if SAVE_PLOTS:
        #print(f"Creating mesh for plots...", end="")
        # mesh = create_mesh_from_cloud(points.cpu().numpy(
        # ), use_top_k=5, distance_threshold=0.125, plot_each_it=-1, subdivisions=6)
        # print("Done.")
        # else:
        mesh = None
        for lr in LRs:
            for loss in LOSSES:
                for encoding in ENCODINGS:
                    for batch_size in BATCH_SIZES:
                        for target_sample_method in TARGET_SAMPLER:
                            for activation in ACTIVATION:
                                run_counter += 1
                                print(
                                    f"\n ---------- RUNNING CONFIG {run_counter} / {TOTAL_RUNS} -------------")
                                print(
                                    f"|LR={lr}\t\t\tloss={loss.__name__}\t\tencoding={encoding.__name__}|")
                                print(
                                    f"|target_sample={target_sample_method}\tactivation={str(activation)[:-2]}\t\tbatch_size={batch_size}|")
                                print(
                                    f"--------------------------------------------")
                                _run_configuration(lr, loss, encoding, batch_size,
                                                   sample, points, masses, target_sample_method, activation, mesh)
        print("###############################################")
        print("#############       SAMPLE DONE     ###########")
        print("###############################################")

    print(f"Writing results csv to {OUTPUT_FOLDER}. \n")
    RESULTS.to_csv(OUTPUT_FOLDER + "/" + "results.csv")
    print("###############################################")
    print("#############   TUTTO FATTO :)    #############")
    print("###############################################")


def _run_configuration(lr, loss_fn, encoding, batch_size, sample, points, masses, target_sample_method, activation, mesh):
    """Runs a specific parameter configur

    Args:
        lr (float): learning rate
        loss_fn (func): Loss function to call
        encoding (func): Encoding function to call
        batch_size (int): Number of target points per batch
        sample (str): Name of the sample to run
        points (torch tensor): Points of the mascon model
        masses (torch tensor): Masses of the mascon model
        target_sample_method (str): Sampling method to use for target points
        activation (Torch fun): Activation function on last network layer
        mesh (pyvista mesh): Mesh of the sample
    """
    # Clear GPU memory
    torch.cuda.empty_cache()

    # Create folder for this specific run
    run_folder = OUTPUT_FOLDER + \
        sample.replace("/", "_") + \
        f"/LR={lr}_loss={loss_fn.__name__}_encoding={encoding.__name__}_" + \
        f"batch_size={batch_size}_target_sample={target_sample_method}_activation={str(activation)[:-2]}/"
    pathlib.Path(run_folder).mkdir(parents=True, exist_ok=True)

    # Init model
    model = init_network(encoding, n_neurons=100, activation=activation)

    # When a new network is created we init empty training logs and we init some loss trend indicators
    loss_log = []
    weighted_average_log = []
    running_loss_log = []
    n_inferences = []
    weighted_average = deque([], maxlen=20)
    running_loss = 1.

    # Here we set the method to sample the target points
    targets_point_sampler = get_target_point_sampler(
        batch_size, method=target_sample_method, radius_bounds=SAMPLE_DOMAIN, scale_bounds=SAMPLE_DOMAIN)

    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=1000, min_lr=5e-6, verbose=False)

    t = tqdm(range(ITERATIONS), ncols=150)
    for it in t:
        # Sample target points
        targets = targets_point_sampler()
        labels = U_L(targets, points, masses)
        # Train
        loss, c = train_on_batch(targets, labels, model, encoding(),
                                 loss_fn, optimizer, scheduler, INTEGRATOR, N_INTEGR_POINTS)

        # Update the loss trend indicators
        running_loss = 0.9 * running_loss + 0.1 * loss.item()
        weighted_average.append(loss.item())

        # Update the logs
        running_loss_log.append(running_loss)
        weighted_average_log.append(np.mean(weighted_average))
        loss_log.append(loss.item())
        n_inferences.append((N_INTEGR_POINTS*batch_size) // 1000)
        wa_out = np.mean(weighted_average)

        t.set_postfix_str(
            f"Loss={loss.item():.3e} | WeightedAvg={wa_out:.3e}\t | c={c:.3e}")

    _save_results(loss_log, running_loss_log,
                  weighted_average_log, model, run_folder)

    if SAVE_PLOTS:
        _save_plots(model, encoding(), mesh, loss_log,
                    running_loss_log, weighted_average_log, n_inferences, run_folder, c)

    # store in results dataframe
    global RESULTS
    RESULTS = RESULTS.append(
        {"Sample": sample, "Type": "ACC" if USE_ACC else "U", "Loss": loss_fn.__name__, "Encoding": encoding.__name__,
         "Integrator": INTEGRATOR.__name__, "Activation": str(activation)[:-2],
         "Batch Size": batch_size, "LR": lr, "Target Sampler": target_sample_method, "Integration Points": N_INTEGR_POINTS,
         "Final Loss": loss_log[-1], "Final Running Loss": running_loss_log[-1], "Final WeightedAvg Loss": weighted_average_log[-1]},
        ignore_index=True
    )


def _make_folders():
    """Creates a folder for each sample that will be run
    """
    global OUTPUT_FOLDER
    pathlib.Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

    for sample in SAMPLES:
        pathlib.Path(OUTPUT_FOLDER + sample.replace("/", "_")
                     ).mkdir(parents=True, exist_ok=True)

    print("Done")


def _load_sample(sample):
    """Loads the mascon model of the sample

    Args:
        sample (str): Sample to load

    Returns:
        torch tensors: points and masses of the sample
    """
    with open(sample, "rb") as file:
        points, masses, name = pk.load(file)

    points = torch.tensor(points)
    masses = torch.tensor(masses)

    print("Name: ", name)
    print("Number of points: ", len(points))
    print("Total mass: ", sum(masses).item())
    print("Maximal minimal distance:", max_min_distance(points))
    return points, masses


def _save_results(loss_log, running_loss_log, weighted_average_log, model, folder):
    """Stores the results of a run

    Args:
        loss_log (list): list of losses recorded
        running_loss_log (list): list of running losses recorded
        weighted_average_log (list): list of weighted average losses recorded
        model (torch model): Torch model that was trained
        folder (str): results folder of the run
    """
    print(f"Saving run results to {folder} ...", end="")
    np.save(folder+"loss_log.npy", loss_log)
    np.save(folder+"running_loss_log.npy", loss_log)
    np.save(folder+"weighted_average_log.npy", loss_log)
    torch.save(model.state_dict(), folder + "model.mdl")
    print("Done.")


def _save_plots(model, encoding, gt_mesh, loss_log, running_loss_log, weighted_average_log, n_inferences, folder, c):
    """Creates plots using the model and stores them

    Args:
        model (torch nn): trained model
        encoding (func): encoding function
        gt_mesh ([type]): ground truth mesh of the sample
        loss_log (list): list of losses recorded
        running_loss_log (list): list of running losses recorded
        weighted_average_log (list): list of weighted average losses recorded
        n_inferences (list): list of number of model evaluations
        folder (str): results folder of the run
    """
    #print("Creating mesh plots...", end="")
    # plot_model_vs_cloud_mesh(model, gt_mesh, encoding,
    #                         save_path=folder + "mesh_plot.pdf")
    # print("Done.")

    print("Creating rejection plot...", end="")
    plot_model_rejection(model, encoding, views_2d=True,
                         bw=True, N=100000, crop_p=0.2, alpha=0.1, s=50, save_path=folder + "rejection_plot.png", c=c)
    print("Done.")

    print("Creating loss plots...", end="")
    plt.figure()
    abscissa = np.cumsum(n_inferences)
    plt.semilogy(abscissa, loss_log)
    plt.semilogy(abscissa, running_loss_log)
    plt.semilogy(abscissa, weighted_average_log)
    plt.xlabel("Thousands of model evaluations")
    plt.ylabel("Loss")
    plt.legend(["Loss", "Running Loss", "Weighted Average Loss"])
    plt.savefig(folder+"loss_plot.png", dpi=150)
    print("Done.")


if __name__ == "__main__":
    run()
