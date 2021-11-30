import pandas as pd
import torch
from tqdm import tqdm
import numpy as np

from ._losses import (
    contrastive_loss,
    normalized_L1_loss,
    normalized_relative_L2_loss,
    normalized_relative_component_loss,
    RMSE,
    relRMSE,
)
from ._mascon_labels import ACC_L
from ._sample_observation_points import get_target_point_sampler
from ._utils import fixRandomSeeds


def validation_mascon(
    mascon_cube_points,
    mascon_cube_masses,
    mascon_points,
    mascon_masses,
    asteroid_pk_path,
    N=5000,
    sampling_altitudes=[0.05, 0.1, 0.25],
    batch_size=100,
    russell_points=3,
    progressbar=True,
):
    """Computes different loss values for the passed model and asteroid with high precision
    Args:
        mascon_cube_points (torch.tensor): cube mascon points
        mascon_cube_masses (torch.tensor): cube mascon masses
        mascon_points (torch.tensor): asteroid mascon points
        mascon_masses (torch.tensor): asteroid mascon masses
        asteroid_pk_path (str): path to the asteroid mesh, necessary for altitude
        mascon_masses_nu (torch.tensor): non-uniform asteroid masses. Pass if using differential training
        N (int, optional): Number of evaluations per altitude. Defaults to 5000.
        sampling_altitudes (np.array, optional): Altitude to sample at for validation. Defaults to [0.05, 0.1, 0.25].
        batch_size (int, optional): batch size (will split N in batches). Defaults to 32.
        russell_points (int , optional): how many points should be sampled per altitude for russel style radial projection sampling. Defaults to 3.
        progressbar (bool, optional): Display a progress. Defaults to True.
    Returns:
        pandas dataframe: Results as df
    """
    torch.cuda.empty_cache()
    fixRandomSeeds()

    loss_fns = [
        normalized_L1_loss,  # normalized_loss, normalized_relative_L2_loss,
        normalized_relative_component_loss,
        RMSE,
        relRMSE,
    ]
    cols = [
        "Altitude",
        "Normalized L1 Loss",  # "Normalized Loss", "Normalized Rel. L2 Loss",
        "Normalized Relative Component Loss",
        "RMSE",
        "relRMSE",
    ]
    results = pd.DataFrame(columns=cols)

    ###############################################
    # Compute validation for radially projected points (outside the asteroid),

    # Low altitude
    torch.cuda.empty_cache()
    pred, labels, loss_values = [], [], []
    target_sampler = get_target_point_sampler(
        russell_points,
        method="radial_projection",
        bounds=[0.0, 0.15625],
        limit_shape_to_asteroid=asteroid_pk_path,
    )

    target_points = target_sampler().detach()

    if progressbar:
        pbar = tqdm(
            desc="Computing validation...",
            total=2 * len(target_points) + N * (len(sampling_altitudes)),
        )

    for idx in range((len(target_points) // batch_size) + 1):
        indices = list(
            range(
                idx * batch_size, np.minimum((idx + 1)
                                             * batch_size, len(target_points))
            )
        )
        points = target_points[indices]
        labels.append(ACC_L(points, mascon_points, mascon_masses).detach())
        pred.append(ACC_L(points, mascon_cube_points,
                          mascon_cube_masses).detach())
        if progressbar:
            pbar.update(batch_size)

    pred = torch.cat(pred)
    labels = torch.cat(labels)

    # Compute Losses
    for loss_fn in loss_fns:
        if loss_fn in [
            contrastive_loss,
            normalized_relative_L2_loss,
            normalized_relative_component_loss,
            RMSE,
            relRMSE,
        ]:
            loss_values.append(torch.mean(
                loss_fn(pred, labels)).cpu().detach().item())
        else:
            loss_values.append(
                torch.mean(loss_fn(pred.view(-1), labels.view(-1)))
                .cpu()
                .detach()
                .item()
            )

    results = results.append(
        dict(zip(cols, ["Low Altitude"] + loss_values)), ignore_index=True
    )

    # High altitude
    torch.cuda.empty_cache()
    pred, labels, loss_values = [], [], []
    target_sampler = get_target_point_sampler(
        russell_points,
        method="radial_projection",
        bounds=[0.15625, 0.3125],
        limit_shape_to_asteroid=asteroid_pk_path,
    )

    target_points = target_sampler().detach()
    for idx in range((len(target_points) // batch_size) + 1):
        indices = list(
            range(
                idx * batch_size, np.minimum((idx + 1)
                                             * batch_size, len(target_points))
            )
        )
        points = target_points[indices]
        labels.append(ACC_L(points, mascon_points, mascon_masses).detach())
        pred.append(ACC_L(points, mascon_cube_points,
                          mascon_cube_masses).detach())
        if progressbar:
            pbar.update(batch_size)

    pred = torch.cat(pred)
    labels = torch.cat(labels)

    # Compute Losses
    for loss_fn in loss_fns:
        if loss_fn in [
            contrastive_loss,
            normalized_relative_L2_loss,
            normalized_relative_component_loss,
            RMSE,
            relRMSE,
        ]:
            loss_values.append(torch.mean(
                loss_fn(pred, labels)).cpu().detach().item())
        else:
            loss_values.append(
                torch.mean(loss_fn(pred.view(-1), labels.view(-1)))
                .cpu()
                .detach()
                .item()
            )

    results = results.append(
        dict(zip(cols, ["High Altitude"] + loss_values)), ignore_index=True
    )

    ################################################
    # Compute errors at different altitudes
    for idx, altitude in enumerate(sampling_altitudes):
        torch.cuda.empty_cache()
        pred, labels, loss_values = [], [], []
        target_sampler = get_target_point_sampler(
            N=batch_size,
            method="altitude",
            bounds=[altitude],
            limit_shape_to_asteroid=asteroid_pk_path,
        )
        for batch in range(N // batch_size):
            target_points = target_sampler().detach()
            labels.append(
                ACC_L(target_points, mascon_points, mascon_masses).detach())
            pred.append(
                ACC_L(target_points, mascon_cube_points,
                      mascon_cube_masses).detach()
            )

            if progressbar:
                pbar.update(batch_size)
            torch.cuda.empty_cache()
        pred = torch.cat(pred)
        labels = torch.cat(labels)

        # Compute Losses
        for loss_fn in loss_fns:
            if loss_fn in [
                contrastive_loss,
                normalized_relative_L2_loss,
                normalized_relative_component_loss,
                RMSE,
                relRMSE,
            ]:
                loss_values.append(
                    torch.mean(loss_fn(pred, labels)).cpu().detach().item()
                )
            else:
                loss_values.append(
                    torch.mean(loss_fn(pred.view(-1), labels.view(-1)))
                    .cpu()
                    .detach()
                    .item()
                )

        results = results.append(
            dict(zip(cols, ["Altitude_" + str(idx)] + loss_values)), ignore_index=True
        )

    if progressbar:
        pbar.close()

    torch.cuda.empty_cache()
    return results
