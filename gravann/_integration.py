import numpy as np
import torch
import sobol_seq
from ._encodings import direct_encoding

import os

# We generate 200000 low-discrepancy points in 3D upon module import and store it as a global
# variable
"""
Sobol low discrepancy sequence in 3 dimensions
"""
sobol_points = sobol_seq.i4_sobol_generate(3, 200000)

# Naive Montecarlo method for the potential


def U_mc(target_points, model, encoding=direct_encoding(), N=3000):
    """Plain Monte Carlo evaluation of the potential from the modelled density

    Args:
        target_points (2-D array-like): a (N,3) array-like object containing the points.
        model (callable (a,b)->1): neural model for the asteroid. 
        encoding: the encoding for the neural inputs.
        N (int): number of points.
    """
    if model[0].in_features != encoding.dim:
        print("encoding is incompatible with the model")
        raise ValueError
    # We generate randomly points in the [-1,1]^3 bounds
    sample_points = torch.rand(N, 3, device=os.environ["TORCH_DEVICE"]) * 2 - 1
    nn_inputs = encoding(sample_points)
    rho = model(nn_inputs)
    retval = torch.empty(len(target_points), 1)
    # Only for the points inside we accumulate the integrand (MC method)
    for i, target_point in enumerate(target_points):
        retval[i] = torch.sum(
            rho/torch.norm(target_point - sample_points, dim=1).view(-1, 1)) / N
    return - 8 * retval

# Low-discrepancy Montecarlo for the potential


def U_ld(target_points, model, encoding=direct_encoding(), N=3000, noise=1e-5):
    """Low discrepancy Monte Carlo evaluation of the potential from the modelled density

    Args:
        target_points (2-D array-like): a (N,3) array-like object containing the points.
        model (callable (a,b)->1): neural model for the asteroid. 
        encoding: the encoding for the neural inputs.
        N (int): number of points.
        noise (float): random noise added to point positions.
    """
    if model[0].in_features != encoding.dim:
        print("encoding is incompatible with the model")
        raise ValueError
    if N > np.shape(sobol_points)[0]:
        print("Too many points the sobol sequence stored in a global variable only contains 200000.")
    # We generate randomly points in the [-1,1]^3 bounds
    if os.environ["TORCH_DEVICE"] != "cpu":
        sample_points = torch.cuda.FloatTensor(
            sobol_points[:N, :] * 2 - 1, device=os.environ["TORCH_DEVICE"]) + torch.rand(N, 3, device=os.environ["TORCH_DEVICE"]) * noise
    else:
        sample_points = torch.tensor(
            sobol_points[:N, :] * 2 - 1) + torch.rand(N, 3) * noise
    nn_inputs = encoding(sample_points)
    rho = model(nn_inputs)
    retval = torch.empty(len(target_points), 1,
                         device=os.environ["TORCH_DEVICE"])
    # Only for the points inside we accumulate the integrand (MC method)
    for i, target_point in enumerate(target_points):
        retval[i] = torch.sum(
            rho/torch.norm(target_point - sample_points, dim=1).view(-1, 1)) / N
    return - 8 * retval

# Trapezoid rule for the potential


def U_trap_opt(target_points, model, encoding=direct_encoding(), N=10000, verbose=False, noise=1e-5):
    """Uses a 3D trapezoid rule for the evaluation of the integral in the potential from the modeled density

    Args:
        target_points (2-D array-like): a (N,3) array-like object containing the points.
        model (callable (a,b)->1): neural model for the asteroid. 
        encoding: the encoding for the neural inputs.
        N (int): number of points.
        verbose (bool, optional): Print intermediate results. Defaults to False.
        noise (float): random noise added to point positions.

    Returns:
        Tensor: Computed potentials per point
    """
    N = int(np.round(np.cbrt(N)))  # approximate subdivisions
    # init result vector
    retval = torch.empty(len(target_points), 1,
                         device=os.environ["TORCH_DEVICE"])

    # Create grid and assemble evaluation points
    grid_1d = torch.linspace(-1, 1, N, device=os.environ["TORCH_DEVICE"])
    h = (grid_1d[1] - grid_1d[0])
    x, y, z = torch.meshgrid(grid_1d, grid_1d, grid_1d)
    eval_points = torch.stack((x.flatten(), y.flatten(), z.flatten())).transpose(
        0, 1).to(os.environ["TORCH_DEVICE"])

    # We add some noise to the evaluated grid points to ensure the networks learns all
    eval_points += torch.rand(N**3, 3,
                              device=os.environ["TORCH_DEVICE"]) * noise

    if verbose:
        print("eval_points=", eval_points)
    if verbose:
        print("h=", h)

    # Evaluate Rho on the grid
    nn_inputs = encoding(eval_points)  # Encode grid
    nn_inputs[nn_inputs != nn_inputs] = 0.0  # set Nans to 0
    rho = model(nn_inputs)

    for i, target_point in enumerate(target_points):

        f_values = rho/torch.norm(target_point -
                                  eval_points, dim=1).view(-1, 1).detach()

        # Evaluate all points
        evaluations = f_values.reshape([N, N, N])  # map to z,y,x
        if verbose:
            print("evaluations=", evaluations)

        # area = h / 2 + (f0 + f2)
        int_x = h / 2 * (evaluations[:, :, 0:-1] + evaluations[:, :, 1:])
        int_x = torch.sum(int_x, dim=2)
        int_y = h / 2 * (int_x[:, 0:-1] + int_x[:, 1:])
        int_y = torch.sum(int_y, dim=1)
        int_z = h / 2 * (int_y[0:-1] + int_y[1:])
        int_z = torch.sum(int_z, dim=0)
        if verbose:
            print("int_x", int_x.shape, int_x)
        if verbose:
            print("int_y", int_y.shape, int_y)
        if verbose:
            print("int_z", int_z.shape, int_z)

        retval[i] = int_z
    return -retval

# Low-discrepancy Montecarlo for the acceleration


def ACC_ld(target_points, model, encoding=direct_encoding(), N=3000, noise=1e-5):
    """Low discrepancy Monte Carlo evaluation of the potential from the modelled density

    Args:
        target_points (2-D array-like): a (N,3) array-like object containing the points.
        model (callable (a,b)->1): neural model for the asteroid. 
        encoding: the encoding for the neural inputs.
        N (int): number of points.
        noise (float): random noise added to point positions.
    """
    # We check that the model is compatible with the encoding in terms of number of inputs
    if model[0].in_features != encoding.dim:
        raise ValueError("encoding is incompatible with the model")
    # We check that there are enough sobol points in the global variable
    if N > np.shape(sobol_points)[0]:
        raise ValueError(
            "Too many points the sobol sequence stored in a global variable only contains 200000.")
    # We generate pseudo-randomly points in the [-1,1]^3 bounds, taking care to have them of the correct type
    if os.environ["TORCH_DEVICE"] != "cpu":
        sample_points = torch.cuda.FloatTensor(
            sobol_points[:N, :] * 2 - 1, device=os.environ["TORCH_DEVICE"]) + torch.rand(N, 3, device=os.environ["TORCH_DEVICE"]) * noise
    else:
        sample_points = torch.tensor(
            sobol_points[:N, :] * 2 - 1) + torch.rand(N, 3) * noise

    # 1 - compute the inputs to the ANN encoding the sampled points
    nn_inputs = encoding(sample_points)
    # 3 - compute the predicted density at the points
    rho = model(nn_inputs)
    retval = torch.empty(len(target_points), 3,
                         device=os.environ["TORCH_DEVICE"])
    # 4 - the mc integral in the hypercube [-1,1]^3 (volume is 8) for each of the target points
    for i, target_point in enumerate(target_points):
        dr = torch.sub(target_point, sample_points)
        retval[i] = torch.sum(
            rho/torch.pow(torch.norm(dr, dim=1), 3).view(-1, 1) * dr, dim=0) / N
    return - 8 * retval
