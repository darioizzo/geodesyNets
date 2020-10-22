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

from gravann import directional_encoding, positional_encoding, direct_encoding, spherical_coordinates
from gravann import normalized_loss, mse_loss
from gravann import ACC_ld, U_mc, U_ld, U_trap_opt, sobol_points
from gravann import U_L
from gravann import enableCUDA, max_min_distance
from gravann import get_target_point_sampler
from gravann import init_network, train_on_batch
from gravann import create_mesh_from_cloud, plot_model_vs_cloud_mesh


os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # Select GPUs
OUTPUT_FOLDER = "results/"                  # Where results will be stored
SAMPLE_PATH = "mascons/"                    # Mascon folder
ITERATIONS = 500                            # Number of training iterations
# SAMPLES = glob(SAMPLE_PATH + "/*.pk")     # Use all available samples
SAMPLES = [                                 # Use some specific samples
    #    "mascons/Eros.pk",
    #    "mascons/Churyumov–Gerasimenko.pk",
    #    "mascons/Itokawa.pk",
    "mascons/sample_01_cluster_2400.pk",
    "mascons/sample_02_cluster_5486.pk"]
N_INTEGR_POINTS = 10000                 # Number of integrations points for U
TARGET_SAMPLER = ["spherical",          # How to sample target points
                  #   "cubical",
                  ]
SAMPLE_DOMAIN = [1.0,                   # Defines the distance of target points
                 1.1]
BATCH_SIZES = [100]                     # For training
LRs = [1e-4]                            # LRs to use
LOSSES = [                              # Losses to use
    normalized_loss,
    #   mse_loss
]

ENCODINGS = [                           # Encodings to test
    directional_encoding,
    # direct_encoding,
    # spherical_coordinates
]
USE_ACC = False                         # Use acceleration instead of U (TODO)
INTEGRATOR = U_trap_opt
ACTIVATION = [                          # Activation function on the last layer
    torch.nn.Sigmoid(),
    torch.nn.Softplus(),
    # torch.nn.Tanh(),
    torch.nn.LeakyReLU(),
]
SAVE_PLOTS = True                       # If plots should be saved.


def run():
    print("Initializing...")

    print("Using the following samples:", SAMPLES)

    # Enable CUDA
    enableCUDA()
    device = os.environ["TORCH_DEVICE"]
    print("Will use device ", device)

    # Make output folders
    print("Making folder structre...", end="")
    _make_folders()

    for sample in SAMPLES:
        print("")
        print(f"--------------- STARTING {sample} ----------------")
        points, masses = _load_sample(sample)

        if SAVE_PLOTS:
            print(f"Creating mesh for plots...", end="")
            mesh = create_mesh_from_cloud(points.cpu().numpy(
            ), use_top_k=5, distance_threshold=0.125, plot_each_it=-1, subdivisions=5)
            print("Done.")
        else:
            mesh = None
        for lr in LRs:
            for loss in LOSSES:
                for encoding in ENCODINGS:
                    for batch_size in BATCH_SIZES:
                        for target_sample_method in TARGET_SAMPLER:
                            for activation in ACTIVATION:
                                print(
                                    f"------------ RUNNING CONFIG ----------------")
                                print(
                                    f"|LR={lr}|\t\tloss={loss.__name__}|\t\tencoding={encoding.__name__}|")
                                print(
                                    f"target_sample={target_sample_method}|\tactivation={str(activation)[:-2]}|\tbatch_size={batch_size}")
                                print(
                                    f"--------------------------------------------")
                                _run_configuration(lr, loss, encoding, batch_size,
                                                   sample, points, masses, target_sample_method, activation, mesh)
        print("###############################################")
        print("#############       SAMPLE DONE     ###########")
        print("###############################################")
    print("###############################################")
    print("#############   TUTTO FATTO :)    #############")
    print("###############################################")


def _run_configuration(lr, loss_fn, encoding, batch_size, sample, points, masses, target_sample_method, activation, mesh):
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

    t = tqdm(range(ITERATIONS))
    for it in t:
        # Sample target points
        targets = targets_point_sampler()
        labels = U_L(targets, points, masses)

        # Train
        loss = train_on_batch(targets, labels, model, encoding(),
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
            f"Loss={loss.item():.8f} | WeightedAvg={wa_out:.8f}\t | Running Loss={running_loss:.8f}")

    _save_results(loss_log, running_loss_log,
                  weighted_average_log, model, run_folder)

    if SAVE_PLOTS:
        _save_plots(model, encoding(), mesh, loss_log,
                    running_loss_log, weighted_average_log, n_inferences, run_folder)


def _make_folders():
    dt_string = datetime.now().strftime("%d_%m_%Y_%H.%M.%S")
    global OUTPUT_FOLDER
    OUTPUT_FOLDER = OUTPUT_FOLDER + "/" + dt_string + "/"
    pathlib.Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

    for sample in SAMPLES:
        pathlib.Path(OUTPUT_FOLDER + sample.replace("/", "_")
                     ).mkdir(parents=True, exist_ok=True)

    print("Done")


def _load_sample(sample):
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
    print(f"Saving run results to {folder} ...", end="")
    np.save(folder+"loss_log.npy", loss_log)
    np.save(folder+"running_loss_log.npy", loss_log)
    np.save(folder+"weighted_average_log.npy", loss_log)
    torch.save(model.state_dict(), folder + "model.mdl")
    print("Done.")


def _save_plots(model, encoding, gt_mesh, loss_log, running_loss_log, weighted_average_log, n_inferences, folder):
    print("Creating plots...")

    print("Creating mesh plots...", end="")
    plot_model_vs_cloud_mesh(model, gt_mesh, encoding,
                             save_path=folder + "mesh_plot.pdf")
    print("Done.")

    # Plot the loss history
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
