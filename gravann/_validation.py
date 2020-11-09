import pandas as pd
import torch
from tqdm import tqdm

from ._losses import contrastive_loss, normalized_loss, normalized_L1_loss
from ._mascon_labels import ACC_L, U_L
from ._sample_observation_points import get_target_point_sampler
from ._integration import ACC_trap, U_trap_opt, compute_integration_grid


def validation(model, encoding, mascon_points, mascon_masses,
               use_acc, asteroid_pk_path, N=5000, N_integration=500000, batch_size=32, progressbar=True):
    """Computes different loss values for the passed model and asteroid with high precision

    Args:
        model (torch.nn): trained model
        encoding (encoding): encoding to use for the points
        mascon_points (torch.tensor): asteroid mascon points
        mascon_masses (torch.tensor): asteroid mascon masses
        use_acc (bool): if acceleration should be used (otherwise potential)
        asteroid_pk_path (str): path to the asteroid mesh, necessary for altitude
        N (int, optional): Number of evaluations per altitude. Defaults to 5000.
        N_integration (int, optional): Number of integrations points to use. Defaults to 500000.
        batch_size (int, optional): batch size (will split N in batches). Defaults to 32.
        progressbar (bool, optional): Display a progress. Defaults to True.

    Returns:
        pandas dataframe: Results as df 
    """
    torch.cuda.empty_cache()
    if use_acc:
        label_function = ACC_L
        integrator = ACC_trap
        integration_grid, h, N_int = compute_integration_grid(N_integration)
    else:
        label_function = U_L
        integrator = U_trap_opt
        integration_grid, h, N_int = compute_integration_grid(N_integration)

    loss_fns = [contrastive_loss, normalized_L1_loss, normalized_loss]
    cols = ["Altitude", "Contrastive Loss",
            "Normalized L1 Loss", "Normalized Loss"]
    results = pd.DataFrame(columns=cols)
    sampling_altitudes = [0.05, 0.1, 0.25]

    if progressbar:
        pbar = tqdm(desc="Computing validation...",
                    total=N * (len(sampling_altitudes) + 1))

    ###############################################
    # Compute validation for random points (outside the asteroid)
    torch.cuda.empty_cache()
    pred, labels, loss_values = [], [], []
    target_sampler = get_target_point_sampler(
        batch_size, method="spherical", bounds=[0, 1], limit_shape_to_asteroid=asteroid_pk_path)
    for batch in range(N // batch_size):
        target_points = target_sampler().detach()
        labels.append(label_function(
            target_points, mascon_points, mascon_masses).detach())

        pred.append(integrator(target_points, model, encoding, N=N_int,
                               h=h, sample_points=integration_grid).detach())

        if progressbar:
            pbar.update(batch_size)
        torch.cuda.empty_cache()

    pred = torch.cat(pred)
    labels = torch.cat(labels)

    # Compute Losses
    for loss_fn in loss_fns:
        if loss_fn == contrastive_loss:
            loss_values.append(torch.mean(
                loss_fn(pred, labels)).cpu().detach().item())
        else:
            loss_values.append(torch.mean(
                loss_fn(pred.view(-1), labels.view(-1))).cpu().detach().item())

    results = results.append(
        dict(zip(cols, ["[0-1] Spherical"] + loss_values)), ignore_index=True)

    ################################################
    # Compute errors at different altitudes
    for altitude in sampling_altitudes:
        torch.cuda.empty_cache()
        pred, labels, loss_values = [], [], []
        target_sampler = get_target_point_sampler(
            N=batch_size, method="altitude",
            bounds=[altitude], limit_shape_to_asteroid=asteroid_pk_path)
        for batch in range(N // batch_size):
            target_points = target_sampler().detach()
            labels.append(label_function(
                target_points, mascon_points, mascon_masses).detach())

            pred.append(integrator(target_points, model, encoding, N=N_int,
                                   h=h, sample_points=integration_grid).detach())

            if progressbar:
                pbar.update(batch_size)
            torch.cuda.empty_cache()
        pred = torch.cat(pred)
        labels = torch.cat(labels)

        # Compute Losses
        for loss_fn in loss_fns:
            if loss_fn == contrastive_loss:
                loss_values.append(torch.mean(
                    loss_fn(pred, labels)).cpu().detach().item())
            else:
                loss_values.append(torch.mean(
                    loss_fn(pred.view(-1), labels.view(-1))).cpu().detach().item())

        results = results.append(
            dict(zip(cols, [altitude] + loss_values)), ignore_index=True)

    if progressbar:
        pbar.close()

    torch.cuda.empty_cache()
    return results
