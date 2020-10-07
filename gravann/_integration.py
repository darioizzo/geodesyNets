import numpy as np
import torch
import sobol_seq
from ._encodings import direct_encoding

"""
Sobol low discrepancy sequence in 3 dimensions
"""
sobol_points = sobol_seq.i4_sobol_generate(3, 200000)

# Naive Montecarlo method


def U_Pmc(target_points, model, encoding=direct_encoding(), N=3000):
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
    sample_points = torch.rand(N, 3) * 2 - 1
    nn_inputs = encoding(sample_points)
    rho = model(nn_inputs)
    retval = torch.empty(len(target_points), 1)
    # Only for the points inside we accumulate the integrand (MC method)
    for i, target_point in enumerate(target_points):
        retval[i] = torch.sum(
            rho/torch.norm(target_point - sample_points, dim=1).view(-1, 1)) / N
    return - 8 * retval

# Low-discrepancy Montecarlo


def U_Pld(target_points, model, encoding=direct_encoding(), N=3000, noise=1e-5):
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
    sample_points = torch.tensor(
        sobol_points[:N, :] * 2 - 1) + torch.rand(N, 3) * noise
    nn_inputs = encoding(sample_points)
    rho = model(nn_inputs)
    retval = torch.empty(len(target_points), 1)
    # Only for the points inside we accumulate the integrand (MC method)
    for i, target_point in enumerate(target_points):
        retval[i] = torch.sum(
            rho/torch.norm(target_point - sample_points, dim=1).view(-1, 1)) / N
    return - 8 * retval
