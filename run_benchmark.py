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
import warnings

from gravann import directional_encoding, positional_encoding, direct_encoding, spherical_coordinates
from gravann import normalized_loss, mse_loss, contrastive_loss, normalized_L1_loss
from gravann import ACC_ld, U_mc, U_ld, U_trap_opt, sobol_points, ACC_trap
from gravann import U_L, ACC_L
from gravann import is_outside
from gravann import enableCUDA, max_min_distance, fixRandomSeeds
from gravann import get_target_point_sampler
from gravann import init_network, train_on_batch
from gravann import create_mesh_from_cloud, plot_model_vs_cloud_mesh, plot_model_rejection, plot_model_vs_mascon_rejection, plot_model_vs_mascon_contours

EXPERIMENT_ID = "run_06_11_2020"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"            # Select GPUs
SAMPLE_PATH = "mascons/"                            # Mascon folder
# Number of training iterations
ITERATIONS = 1000
# SAMPLES = glob(SAMPLE_PATH + "/*.pk")             # Use all available samples
SAMPLES = [                                         # Use some specific samples
    # "Eros.pk",
    # "Churyumov-Gerasimenko.pk",
    # "Itokawa_non_uniform.pk",
    # "Bennu_lp.pk",
    "sample_01_cluster_2400.pk",
    # "sample_02_cluster_5486.pk",
    # "sample_03_cluster_2284",
    # "sample_04_cluster_6674_hollow_0.3_0.3.pk",
    # "sample_04_cluster_7315",
    # "sample_06_cluster_6137.pk",
    # "sample_07_cluster_2441.pk",
    # "sample_08_cluster_1970.pk",
    # "sample_09_cluster_1896.pk"
]

N_INTEGR_POINTS = 30000                # Number of integrations points for U
TARGET_SAMPLER = ["spherical",      # How to sample target points
                  # "cubical",
                  ]
SAMPLE_DOMAIN = [0.0,                   # Defines the distance of target points
                 1]
BATCH_SIZES = [100]                    # For training
LRs = [1e-4]                            # LRs to use
LOSSES = [                              # Losses to use
    mse_loss,
    # normalized_loss,
    # normalized_L1_loss,
    # contrastive_loss
]

ENCODINGS = [                           # Encodings to test
    # directional_encoding(),
    direct_encoding(),
    # positional_encoding(3),
    # spherical_coordinates()
]
USE_ACC = True                         # Use acceleration instead of U
if USE_ACC:
    INTEGRATOR = ACC_trap
    EXPERIMENT_ID = EXPERIMENT_ID + "_" + "ACC"
else:
    INTEGRATOR = U_trap_opt
    EXPERIMENT_ID = EXPERIMENT_ID + "_" + "U"


MODEL_TYPE = "default"  # either "siren", "default", "nerf"
EXPERIMENT_ID = EXPERIMENT_ID + "_" + MODEL_TYPE

# We can now name the output folder
OUTPUT_FOLDER = "results/" + EXPERIMENT_ID + "/"    # Results folder


ACTIVATION = [                          # Activation function on the last layer (only for NERF)
    torch.nn.Sigmoid(),
    # torch.nn.Softplus(),
    # torch.nn.Tanh(),
    # torch.nn.LeakyReLU(),
]
if len(ACTIVATION) > 1 and MODEL_TYPE == "siren":
    warnings.warn(
        "Different activation functions are not compatible with Siren network.")


SAVE_PLOTS = True                       # If plots should be saved.
PLOTTING_POINTS = 2500                  # Points per rejection plot

RESULTS = pd.DataFrame(columns=["Sample", "Type", "Model", "Loss", "Encoding", "Integrator", "Activation",
                                "Batch Size", "LR", "Target Sampler", "Integration Points", "Final Loss", "Final WeightedAvg Loss"])


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
                                    f"|LR={lr}\t\t\tloss={loss.__name__}\t\tencoding={encoding.name}|")
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
    global RESULTS
    if os.path.isfile(OUTPUT_FOLDER + "/" + "results.csv"):
        previous_results = pd.read_csv(OUTPUT_FOLDER + "/" + "results.csv")
        RESULTS = pd.concat([previous_results, RESULTS])
    RESULTS.to_csv(OUTPUT_FOLDER + "/" + "results.csv", index=False)
    print("###############################################")
    print("#############   TUTTO FATTO :)    #############")
    print("###############################################")


def _run_configuration(lr, loss_fn, encoding, batch_size, sample, mascon_points, mascon_masses, target_sample_method, activation, mesh):
    """Runs a specific parameter configur

    Args:
        lr (float): learning rate
        loss_fn (func): Loss function to call
        encoding (func): Encoding function to call
        batch_size (int): Number of target points per batch
        sample (str): Name of the sample to run
        mascon_points (torch tensor): Points of the mascon model
        mascon_masses (torch tensor): Masses of the mascon model
        target_sample_method (str): Sampling method to use for target points
        activation (Torch fun): Activation function on last network layer
        mesh (pyvista mesh): Mesh of the sample
    """
    # Clear GPU memory
    torch.cuda.empty_cache()

    # Fix the random seeds for this run
    fixRandomSeeds()

    # Create folder for this specific run
    run_folder = OUTPUT_FOLDER + \
        sample.replace("/", "_") + \
        f"/LR={lr}_loss={loss_fn.__name__}_encoding={encoding.name}_" + \
        f"batch_size={batch_size}_target_sample={target_sample_method}_activation={str(activation)[:-2]}/"
    pathlib.Path(run_folder).mkdir(parents=True, exist_ok=True)

    # Init model
    model = init_network(encoding, n_neurons=100,
                         activation=activation, model_type=MODEL_TYPE)

    # When a new network is created we init empty training logs and we init some loss trend indicators
    loss_log = []
    weighted_average_log = []
    n_inferences = []
    weighted_average = deque([], maxlen=20)

    # Here we set the method to sample the target points
    targets_point_sampler = get_target_point_sampler(
        batch_size, method=target_sample_method, bounds=SAMPLE_DOMAIN, limit_shape_to_asteroid="3dmeshes/" + sample)

    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=1000, min_lr=5e-6, verbose=False)

    # Sample target points
    target_points = targets_point_sampler()

    t = tqdm(range(ITERATIONS), ncols=150)
    for it in t:
        # Each ten epochs we resample the target points
        if (it % 10 == 0):
            target_points = targets_point_sampler()
        # We generate the labels
        if USE_ACC:
            labels = ACC_L(target_points, mascon_points, mascon_masses)
        else:
            labels = U_L(target_points, mascon_points, mascon_masses)

        # Train
        loss, c = train_on_batch(target_points, labels, model, encoding,
                                 loss_fn, optimizer, scheduler, INTEGRATOR, N_INTEGR_POINTS)

        # Update the loss trend indicators
        weighted_average.append(loss.item())

        # Update the logs
        weighted_average_log.append(np.mean(weighted_average))
        loss_log.append(loss.item())
        n_inferences.append((N_INTEGR_POINTS*batch_size) // 1000)
        wa_out = np.mean(weighted_average)

        t.set_postfix_str(
            f"Loss={loss.item():.3e} | WeightedAvg={wa_out:.3e}\t | c={c:.3e}")
        # Each hundred epochs we produce the plots
        if (it % 100 == 0):
            # Save a plot
            plot_model_rejection(model, encoding, views_2d=True, bw=True, N=PLOTTING_POINTS, alpha=0.1,
                                 s=50, save_path=run_folder + "rejection_plot_iter" + format(it, '06d') + ".png", c=c, progressbar=False)
            plot_model_vs_mascon_contours(model, encoding, mascon_points, N=PLOTTING_POINTS,
                                          save_path=run_folder + "contour_plot_iter" + format(it, '06d') + ".png", c=c)
            plt.close('all')

    _save_results(loss_log, weighted_average_log, model, run_folder)

    if SAVE_PLOTS:
        _save_plots(model, encoding, mascon_points, mesh, loss_log,
                    weighted_average_log, n_inferences, run_folder, c)

    # store in results dataframe
    global RESULTS
    RESULTS = RESULTS.append(
        {"Sample": sample, "Type": "ACC" if USE_ACC else "U", "Model": MODEL_TYPE,  "Loss": loss_fn.__name__, "Encoding": encoding.name,
         "Integrator": INTEGRATOR.__name__, "Activation": str(activation)[:-2],
         "Batch Size": batch_size, "LR": lr, "Target Sampler": target_sample_method, "Integration Points": N_INTEGR_POINTS,
         "Final Loss": loss_log[-1], "Final WeightedAvg Loss": weighted_average_log[-1]},
        ignore_index=True
    )

    # store run config
    cfg = {"Sample": sample, "Type": "ACC" if USE_ACC else "U", "Model": MODEL_TYPE,  "Loss": loss_fn.__name__, "Encoding": encoding.name,
           "Integrator": INTEGRATOR.__name__, "Activation": str(activation)[:-2],
           "Batch Size": batch_size, "LR": lr, "Target Sampler": target_sample_method, "Integration Points": N_INTEGR_POINTS}
    with open(run_folder+'config.pk', 'wb') as handle:
        pk.dump(cfg, handle)


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
    with open("mascons/" + sample, "rb") as file:
        points, masses, name = pk.load(file)

    points = torch.tensor(points)
    masses = torch.tensor(masses)

    print("Name: ", name)
    print("Number of points: ", len(points))
    print("Total mass: ", sum(masses).item())
    print("Maximal minimal distance:", max_min_distance(points))
    return points, masses


def _save_results(loss_log, weighted_average_log, model, folder):
    """Stores the results of a run

    Args:
        loss_log (list): list of losses recorded
        weighted_average_log (list): list of weighted average losses recorded
        model (torch model): Torch model that was trained
        folder (str): results folder of the run
    """
    print(f"Saving run results to {folder} ...", end="")
    np.save(folder+"loss_log.npy", loss_log)
    np.save(folder+"weighted_average_log.npy", loss_log)
    torch.save(model.state_dict(), folder + "model.mdl")
    print("Done.")


def _save_plots(model, encoding, mascon_points, gt_mesh, loss_log, weighted_average_log, n_inferences, folder, c):
    """Creates plots using the model and stores them

    Args:
        model (torch nn): trained model
        encoding (func): encoding function
        mascon_points (torch tensor): Points of the mascon model
        gt_mesh ([type]): ground truth mesh of the sample
        loss_log (list): list of losses recorded
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
                         bw=True, N=PLOTTING_POINTS, alpha=0.1, s=50, save_path=folder + "rejection_plot_iter_9.png", c=c)
    print("Done.")
    print("Creating model_vs_mascon_rejection plot...", end="")
    plot_model_vs_mascon_rejection(
        model, encoding, mascon_points, N=PLOTTING_POINTS, save_path=folder + "model_vs_mascon_rejection.png", c=c)
    print("Done.")

    print("Creating model_vs_mascon_contours plot...", end="")
    plot_model_vs_mascon_contours(
        model, encoding, mascon_points, N=PLOTTING_POINTS, save_path=folder + "contour_plot_iter_9.png", c=c)
    print("Done.")

    print("Creating loss plots...", end="")
    plt.figure()
    abscissa = np.cumsum(n_inferences)
    plt.semilogy(abscissa, loss_log)
    plt.semilogy(abscissa, weighted_average_log)
    plt.xlabel("Thousands of model evaluations")
    plt.ylabel("Loss")
    plt.legend(["Loss", "Weighted Average Loss"])
    plt.savefig(folder+"loss_plot.png", dpi=150)
    print("Done.")


if __name__ == "__main__":
    run()
