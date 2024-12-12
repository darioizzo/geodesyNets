from torch import nn
import torch
import pathlib
import pandas as pd
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import pickle as pk
import time

from ._losses import contrastive_loss, zero_L1_loss, normalized_relative_L2_loss, normalized_relative_component_loss
from ._mascon_labels import ACC_L, ACC_L_differential, U_L
from ._sample_observation_points import get_target_point_sampler
from ._io import load_sample, save_results, save_plots
from ._plots import plot_model_rejection, plot_model_vs_mascon_contours
from ._utils import fixRandomSeeds, EarlyStopping
from ._validation import validation, validation_results_unpack_df

from .networks._siren import Siren
from .networks._siren_q import SirenQ
from .networks._nerf import NERF

# Required for loading runs
from .networks._abs_layer import AbsLayer
from ._encodings import *


def _weights_init(m):
    """Network initialization scheme (note that if xavier uniform is used all outputs will start at, roughly 0.5)

    Args:
        m (torch layer): layer to initialize
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.uniform_(m.bias.data, -0.0, 0.0)


def init_network(encoding, n_neurons=100, activation=nn.Sigmoid(), model_type="default", siren_omega=30., hidden_layers=9):
    """ Network architecture. Note that the dimensionality of the first linear layer must match the output of the encoding chosen

    Args:
        encoding (encoding): encoding instance to use for the network
        n_neurons (int, optional): Number of neurons per layer. Defaults to 100.
        activation (torch activation function, optional): Activation function for the last network layer. Defaults to nn.Sigmoid().
        model_type (str,optional): Defines what model to use. Available "siren", "default", "nerf". Defaults to "default".
        siren_omega (float,optional): Omega value for siren activations. Defaults to 30.
        hidden_layers (int, optional): Number of hidden layers in the network. Defaults to 9.

    Returns:
        torch model: Initialized model
    """
    if model_type == "default":
        modules = []

        # input layer
        modules.append(nn.Linear(encoding.dim, n_neurons))
        modules.append(nn.ReLU())

        # hidden layers
        for _ in range(hidden_layers-1):
            modules.append(nn.Linear(n_neurons, n_neurons))
            modules.append(nn.ReLU())

        # final layer
        modules.append(nn.Linear(n_neurons, 1))
        modules.append(activation)
        model = nn.Sequential(*modules)

        # Applying our weight initialization
        _ = model.apply(_weights_init)
        model.in_features = encoding.dim
        return model
    elif model_type == "nerf":
        return NERF(in_features=encoding.dim, n_neurons=n_neurons, activation=activation, skip=[4], hidden_layers=hidden_layers)
    elif model_type == "siren":
        return Siren(in_features=encoding.dim, out_features=1, hidden_features=n_neurons,
                     hidden_layers=hidden_layers, outermost_linear=True, outermost_activation=activation,
                     first_omega_0=siren_omega, hidden_omega_0=siren_omega)
    elif model_type == "siren_q":
        return SirenQ(in_features=encoding.dim, out_features=1, hidden_features=n_neurons,
                     hidden_layers=hidden_layers, outermost_linear=True, outermost_activation=activation,
                     first_omega_0=siren_omega, hidden_omega_0=siren_omega)


def train_on_batch(targets, labels, model, encoding, loss_fn, optimizer, scheduler, integrator, N, vision_targets=None, integration_domain=None):
    """Trains the passed model on the passed batch

    Args:
        targets (tensor): target points for training
        labels (tensor): labels at the target points
        model (torch model): model to train
        encoding (func): encoding function for the model
        loss_fn (func): loss function for training
        optimizer (torch optimizer): torch optimizer to use
        scheduler (torch LR scheduler): torch LR scheduler to use
        integrator (func): integration function to call for the training loss
        N (int): Number of integration points to use for training
        vision_targets (torch.tensor): If not None will eval L1 loss assuming that density at this points should be 0
        integration_domain (torch.tensor): Domain to pick integration points in, only works with trapezoid for now

    Returns:
        torch tensor: losses
    """
    # Compute the loss (use N=3000 to start with, then, eventually, beef it up to 200000)
    predicted = integrator(targets, model, encoding,
                           N=N, domain=integration_domain)
    c = torch.sum(predicted*labels)/torch.sum(predicted*predicted)
    if loss_fn == contrastive_loss or loss_fn == normalized_relative_L2_loss or loss_fn == normalized_relative_component_loss:
        loss = loss_fn(predicted, labels)
    else:
        loss = loss_fn(predicted.view(-1), labels.view(-1))

    # Urge points outside asteroid to have 0 density.
    vision_loss = torch.tensor([0.0])
    if vision_targets is not None:
        encoded_vision_targets = encoding(vision_targets)
        predictions_at_vision_targets = model(encoded_vision_targets)
        vision_loss = torch.mean(zero_L1_loss(c*predictions_at_vision_targets))
        loss += vision_loss

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()

    # Perform a step in LR scheduler to update LR
    scheduler.step(loss.item())

    return loss, c, vision_loss


def _init_training_run(cfg, sample, lr, loss_fn, encoding, batch_size, target_sample_method, activation, omega, hidden_layers, n_neurons):
    """Initializes params for the training run

    Args:
        cfg (dict): global run cfg 
        sample (str): Sample to load
        lr (float): learning rate
        loss_fn (func): Loss function to call
        encoding (func): Encoding function to call
        batch_size (int): Number of target points per batch
        target_sample_method (str): Sampling method to use for target points
        activation (Torch fun): Activation function on last network layer
        omega (float): Siren omega value
        hidden_layers (int, optional): Number of hidden layers in the network.
        n_neurons (int, optional): Number of neurons per layer.

    Returns:
        model,early_stopper,optimizer,scheduler, target_sampler,vis_sampler,run_folder
    """
    # Clear GPU memory
    torch.cuda.empty_cache()

    # Fix the random seeds for this run
    fixRandomSeeds()

    # Create folder for this specific run
    run_folder = cfg["output_folder"] + \
        sample.replace("/", "_") + \
        f"/LR={lr}_loss={loss_fn.__name__}_ENC={encoding.name}_" + \
        f"BS={batch_size}_target_sample={target_sample_method}_ACT={str(activation)[:-2]}_omega={omega:.2}" + \
        f"_layers={hidden_layers}_neurons={n_neurons}/"
    pathlib.Path(run_folder).mkdir(parents=True, exist_ok=True)

    early_stopper = EarlyStopping(save_folder=run_folder)

    # Init model
    model = init_network(encoding, n_neurons=n_neurons,
                         activation=activation, model_type=cfg["model"]["type"], siren_omega=omega, hidden_layers=hidden_layers)

    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=200, min_lr=1e-6, verbose=True)

    # Here we set the method to sample the target points. We use a low precision mesh to exclude points inside the asteroid.
    sample_lp = sample[:-3]+"_lp.pk"
    targets_point_sampler = get_target_point_sampler(
        batch_size, method=target_sample_method,
        bounds=cfg["model"]["sample_domain"], limit_shape_to_asteroid="3dmeshes/" + sample_lp)

    if cfg["training"]["visual_loss"]:
        # Setup sampler to get points outside asteroid for visual loss
        visual_target_points_sampler = get_target_point_sampler(batch_size, method="cubical", bounds=[
            0.0, 1.0], limit_shape_to_asteroid="3dmeshes/" + sample_lp)
    else:
        def visual_target_points_sampler(): return None

    return model, early_stopper, optimizer, scheduler, targets_point_sampler, visual_target_points_sampler, run_folder


def run_training(cfg, sample, loss_fn, encoding, batch_size, target_sample_method, activation, omega, hidden_layers, n_neurons):
    """Runs a specific parameter configuration
    Args:
        cfg (dict): global run cfg 
        sample (str): sample to load
        loss_fn (func): Loss function to call
        encoding (func): Encoding function to call
        batch_size (int): Number of target points per batch
        target_sample_method (str): Sampling method to use for target points
        activation (Torch fun): Activation function on last network layer
        omega (float): Siren omega value
        hidden_layers (int, optional): Number of hidden layers in the network.
        n_neurons (int, optional): Number of neurons per layer.
    """
    start = time.time()
    # Initialize everything we need
    initialized_vars = _init_training_run(
        cfg, sample, cfg["training"]["lr"], loss_fn, encoding, batch_size, target_sample_method, activation, omega, hidden_layers, n_neurons)
    model, early_stopper, optimizer, scheduler, targets_point_sampler, visual_target_points_sampler, run_folder = initialized_vars

    mascon_points, mascon_masses_u, mascon_masses_nu = load_sample(
        sample, cfg["training"]["differential_training"])

    # When a new network is created we init empty training logs and we init some loss trend indicators
    loss_log, lr_log, vision_loss_log, weighted_average_log, n_inferences = [], [], [], [], []
    weighted_average = deque([], maxlen=20)

    t = tqdm(range(cfg["training"]["iterations"]), ncols=150)
    # At the beginning (first plots) we assume no learned c
    c = 1.
    for it in t:
        # Each hundred epochs we produce the plots
        if (it % 100 == 0):
            # Save a plot
            plot_model_rejection(model, encoding, views_2d=True, bw=True, N=cfg["plotting_points"], alpha=0.1,
                                 s=50, save_path=run_folder + "rejection_plot_iter" + format(it, '06d') + ".png", c=c, progressbar=False)
            plot_model_vs_mascon_contours(model, encoding, mascon_points, N=cfg["plotting_points"],
                                          save_path=run_folder + "contour_plot_iter" + format(it, '06d') + ".png", c=c)
            plt.close('all')
        # Each ten epochs we resample the target points
        if (it % 10 == 0):
            target_points = targets_point_sampler()

        # will be None if not USE_VISUAL_LOSS
        visual_target_points = visual_target_points_sampler()

        # We generate the labels
        if cfg["model"]["use_acceleration"]:
            if cfg["training"]["differential_training"]:
                labels = ACC_L_differential(
                    target_points, mascon_points, mascon_masses_u, mascon_masses_nu)
            else:
                labels = ACC_L(target_points, mascon_points, mascon_masses_u)
        else:
            labels = U_L(target_points, mascon_points, mascon_masses_u)

        # Train
        loss, c, vision_loss = train_on_batch(target_points, labels, model, encoding,
                                              loss_fn, optimizer, scheduler, cfg[
                                                  "integrator"], cfg["integration"]["points"],
                                              vision_targets=visual_target_points, integration_domain=cfg["integration"]["domain"])

        # Update the loss trend indicators
        weighted_average.append(loss.item())

        # Update the logs
        lr_log.append(optimizer.param_groups[0]['lr'])
        weighted_average_log.append(np.mean(weighted_average))
        loss_log.append(loss.item())
        vision_loss_log.append(vision_loss.item())
        n_inferences.append((cfg["integration"]["points"]*batch_size) // 1000)
        wa_out = np.mean(weighted_average)

        t.set_postfix_str(
            f"L={loss.item():.3e} | AvgL={wa_out:.3e} | c={c:.3e} | visionL={vision_loss.item():.3e}")

        if early_stopper.early_stop(loss.item(), model):
            print(
                f"Early stopping at minimal loss {early_stopper.minimal_loss}")
            break

        torch.cuda.empty_cache()
    # Restore best checkpoint
    print("Restoring best checkpoint for validation...")
    model.load_state_dict(torch.load(run_folder + "best_model.mdl"))
    validation_results = validation(
        model, encoding, mascon_points, mascon_masses_u,
        cfg["model"]["use_acceleration"], "3dmeshes/" + sample,
        mascon_masses_nu=mascon_masses_nu,
        N_integration=500000, N=cfg["training"]["validation_points"])

    save_results(loss_log, weighted_average_log,
                 validation_results, model, run_folder)

    save_plots(model, encoding, mascon_points, lr_log, loss_log,
               weighted_average_log, vision_loss_log,  n_inferences, run_folder, c, cfg["plotting_points"])

    # store run config
    cfg_dict = {"Sample": sample, "Type": "ACC" if cfg["model"]["use_acceleration"] else "U", "Model": cfg["model"]["type"],  "Loss": loss_fn.__name__, "Encoding": encoding.name,
                "Integrator": cfg["integrator"].__name__, "Activation": str(activation)[:-2],
                "n_neurons": cfg["model"]["n_neurons"], "hidden_layers": cfg["model"]["hidden_layers"],
                "Batch Size": batch_size, "LR": cfg["training"]["lr"], "Target Sampler": target_sample_method,
                "Integration Points": cfg["integration"]["points"], "Vision Loss": cfg["training"]["visual_loss"], "c": c}

    with open(run_folder+'config.pk', 'wb') as handle:
        pk.dump(cfg_dict, handle)

    # Compute validation results
    val_res = validation_results_unpack_df(validation_results)

    end = time.time()
    runtime = end - start
    result_dictionary = {"Sample": sample,
                         "Type": "ACC" if cfg["model"]["use_acceleration"] else "U", "Model": cfg["model"]["type"],  "Loss": loss_fn.__name__, "Encoding": encoding.name,
                         "Integrator": cfg["integrator"].__name__, "Activation": str(activation)[:-2],
                         "Batch Size": batch_size, "LR": cfg["training"]["lr"], "Target Sampler": target_sample_method, "Integration Points": cfg["integration"]["points"],
                         "Runtime": runtime, "Final Loss": loss_log[-1], "Final WeightedAvg Loss": weighted_average_log[-1], "Final Vision Loss": vision_loss_log[-1]}
    results_df = pd.concat(
        [pd.DataFrame([result_dictionary]), val_res], axis=1)
    return results_df


def load_model_run(folderpath, differential_training=False):
    """Will load a model, sample and cfg from a saved training run.

    Args:
        folderpath (str): Path to the folder containing the config.pk and model.mdl
        differential_training (bool): Indicates if differential training was used. Defaults to False.

    Returns:
        model, encoding, sample, c, use_acc, mascon_points, mascon_masses_u, mascon_masses_nu
    """
    # Load run parameters
    with open(folderpath + "config.pk", "rb") as file:
        params = pk.load(file)

    sample = params["Sample"]
    encoding = globals()[params["Encoding"]]()
    omega = float(folderpath.split("omega=")[1].split("/")[0].split("_")[0])

    if params["Activation"] == "AbsLayer":
        activation = AbsLayer()
    else:
        activation = getattr(torch.nn, params["Activation"])()

    if "n_neurons" in params and "hidden_layers" in params:  # newer cfgs have these entries
        model = init_network(
            encoding, model_type=params["Model"], activation=activation, n_neurons=params["n_neurons"], hidden_layers=params["hidden_layers"], siren_omega=omega)
    else:
        model = init_network(
            encoding, model_type=params["Model"], activation=activation, siren_omega=omega)
    model.load_state_dict(torch.load(folderpath + "best_model.mdl"))

    # if not differential, _nu masses will just be None
    mascon_points, mascon_masses_u, mascon_masses_nu = load_sample(
        sample, use_differential=differential_training)

    c = params["c"].item()

    use_acc = True if params["Type"] == "ACC" else False

    print("Successfully loaded")
    print(params)

    return model, encoding, sample, c, use_acc, mascon_points, mascon_masses_u, mascon_masses_nu, params
