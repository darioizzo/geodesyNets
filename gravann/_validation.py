import pandas as pd
import torch
from tqdm import tqdm

from ._losses import contrastive_loss, normalized_loss, normalized_L1_loss, normalized_relative_L2_loss, normalized_relative_component_loss
from ._mascon_labels import ACC_L, U_L, ACC_L_non_uniform
from ._sample_observation_points import get_target_point_sampler
from ._integration import ACC_trap, U_trap_opt, compute_integration_grid


def compute_c_for_model(model, encoding, mascon_points, mascon_masses, use_acc):
    """Computes the current c constant for a model.

    Args:
        model (torch.nn): trained model
        encoding (encoding): encoding to use for the points
        mascon_points (torch.tensor): asteroid mascon points
        mascon_masses (torch.tensor): asteroid mascon masses
        use_acc (bool): if acceleration should be used (otherwise potential)
    """
    targets_point_sampler = get_target_point_sampler(
        500, method="spherical", bounds=[1.0, 1.1])
    target_points = targets_point_sampler()
    if use_acc:
        labels = ACC_L(target_points, mascon_points, mascon_masses)
        predicted = ACC_trap(target_points, model, encoding, N=100000)
    else:
        labels = U_L(target_points, mascon_points, mascon_masses)
        predicted = U_trap_opt(target_points, model, encoding, N=100000)
    return (torch.sum(predicted*labels)/torch.sum(predicted*predicted)).item()


def validation_results_df_to_string(validation_results):
    result_strings = {}
    for idx, val in validation_results.iterrows():
        result_strings[val["Altitude"]
                       ] = f'ContrL={val["Contrastive Loss"]:.3e},NormL1={val["Normalized L1 Loss"]:.3e},NormL2={val["Normalized Loss"]:.3e},RelL2={val["Normalized Rel. L2 Loss"]:.3e},RelComponent={val["Normalized Relative Component Loss"]}'

    return result_strings


def validation(model, encoding, mascon_points, mascon_masses,
               use_acc, asteroid_pk_path,  mascon_masses_nu=None, N=5000, N_integration=500000, batch_size=100, progressbar=True):
    """Computes different loss values for the passed model and asteroid with high precision

    Args:
        model (torch.nn): trained model
        encoding (encoding): encoding to use for the points
        mascon_points (torch.tensor): asteroid mascon points
        mascon_masses (torch.tensor): asteroid mascon masses
        use_acc (bool): if acceleration should be used (otherwise potential)
        asteroid_pk_path (str): path to the asteroid mesh, necessary for altitude
        mascon_masses_nu (torch.tensor): non-uniform asteroid masses. Pass if using differential training
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
    if mascon_masses_nu is not None:
        def label_function(tp, mp, mm): return ACC_L_non_uniform(
            tp, mp, mm, mascon_masses_nu)

    loss_fns = [contrastive_loss, normalized_L1_loss,
                normalized_loss, normalized_relative_L2_loss, normalized_relative_component_loss]
    cols = ["Altitude", "Contrastive Loss",
            "Normalized L1 Loss", "Normalized Loss", "Normalized Rel. L2 Loss", "Normalized Relative Component Loss"]
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
        if loss_fn == contrastive_loss or loss_fn == normalized_relative_L2_loss or loss_fn == normalized_relative_component_loss:
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
            if loss_fn == contrastive_loss or loss_fn == normalized_relative_L2_loss or loss_fn == normalized_relative_component_loss:
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
