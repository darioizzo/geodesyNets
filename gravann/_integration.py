import numpy as np
import torch
import warnings
import sobol_seq
from ._encodings import direct_encoding

import os


def compute_sobol_points(N=3,M=500000):
    """Generates the low-discrpancy points Sobol sequence

    Args:
        N (int, optional): Dimension. Defaults to 3.
        M (int, optional): Number of points. Defaults to 500000.

    Returns:
        array NxM: the points.
    """
    return sobol_seq.i4_sobol_generate(N, M)

# Naive Montecarlo method for the potential


def rho_trap(model, encoding, mask, c, N=10000):
    """Compute integral over the model where points matching some mask are set to 0

    Args:
        model (callable (a,b)->1): neural model for the asteroid. 
        encoding: the encoding for the neural inputs.
        mask (func): a function that given [N,3] returns a [N] torch.tensor bool as mask
        c (float): scaling factor for model rho
        N (int, optional): Integration grid points. Defaults to 10000.

    Returns:
        torch.tensor: integral value
    """
    # clean up memory to ensure we have space
    torch.cuda.empty_cache()

    # create integration grid
    sample_points, h, N = compute_integration_grid(N)

    # Evaluate Rho on the grid
    rho = _compute_model_output(
        model, encoding, sample_points) * c

    rho[mask(sample_points)] = 0.0

    evaluations = rho.reshape([N, N, N])  # map to z,y,x

    # area = h / 2 * (f0 + f2)
    int_x = h[0] / 2 * (evaluations[:, :, 0:-1] + evaluations[:, :, 1:])
    int_x = torch.sum(int_x, dim=2)
    int_y = h[1] / 2 * (int_x[:, 0:-1] + int_x[:, 1:])
    int_y = torch.sum(int_y, dim=1)
    int_z = h[2] / 2 * (int_y[0:-1] + int_y[1:])
    int_z = torch.sum(int_z, dim=0)

    return int_z


def U_mc(target_points, model, encoding=direct_encoding(), N=3000, domain=None):
    """Plain Monte Carlo evaluation of the potential from the modelled density

    Args:
        target_points (2-D array-like): a (N,3) array-like object containing the points.
        model (callable (a,b)->1): neural model for the asteroid. 
        encoding: the encoding for the neural inputs.
        N (int): number of points.
        domain (torch.tensor): integration domain [3,2] , pass None for [-1,1]^3. Currently Not Implemented!
    """
    if domain is not None:
        raise NotImplementedError(
            "Custom domain is not yet implemented for U_mc.")

    # init result vector
    retval = torch.empty(len(target_points), 1)

    # We generate randomly points in the [-1,1]^3 bounds
    sample_points = torch.rand(N, 3, device=os.environ["TORCH_DEVICE"]) * 2 - 1

    # Evaluate Rho on the points
    rho = _compute_model_output(model, encoding, sample_points)

    # Only for the points inside we accumulate the integrand (MC method)
    for i, target_point in enumerate(target_points):
        retval[i] = torch.sum(
            rho/torch.norm(target_point - sample_points, dim=1).view(-1, 1)) / N
    return - 8 * retval

# Low-discrepancy Montecarlo for the potential


def U_ld(target_points, model, sobol_points, encoding=direct_encoding(), N=3000, noise=0., domain=None):
    """Low discrepancy Monte Carlo evaluation of the potential from the modelled density

    Args:
        target_points (2-D array-like): a (N,3) array-like object containing the points.
        model (callable (a,b)->1): neural model for the asteroid. 
        encoding: the encoding for the neural inputs.
        sobol_points: the Sobol points
        N (int): number of points.
        noise (float): random noise added to point positions.
        domain (torch.tensor): integration domain [3,2] , pass None for [-1,1]^3. Currently Not Implemented!
    """
    if domain is not None:
        raise NotImplementedError(
            "Custom domain is not yet implemented for U_ld.")

    # init result vector
    retval = torch.empty(len(target_points), 1,
                         device=os.environ["TORCH_DEVICE"])

    if N > np.shape(sobol_points)[0]:
        print("Too many points queried, the Sobol points passed are less.")
    # We generate randomly points in the [-1,1]^3 bounds
    if os.environ["TORCH_DEVICE"] != "cpu":
        sample_points = torch.cuda.FloatTensor(
            sobol_points[:N, :] * 2 - 1, device=os.environ["TORCH_DEVICE"]) + torch.rand(N, 3, device=os.environ["TORCH_DEVICE"]) * noise
    else:
        sample_points = torch.tensor(
            sobol_points[:N, :] * 2 - 1) + torch.rand(N, 3) * noise

    # Evaluate Rho on the grid
    rho = _compute_model_output(model, encoding, sample_points)

    # Compute the integral using the sampled and target points
    for i, target_point in enumerate(target_points):
        retval[i] = torch.sum(
            rho/torch.norm(target_point - sample_points, dim=1).view(-1, 1)) / N
    return - 8 * retval

# Trapezoid rule for the potential


def U_trap_opt(target_points, model, encoding=direct_encoding(), N=10000, verbose=False, noise=0., sample_points=None, h=None, domain=None):
    """Uses a 3D trapezoid rule for the evaluation of the integral in the potential from the modeled density

    Args:
        target_points (2-D array-like): a (N,3) array-like object containing the points.
        model (callable (a,b)->1): neural model for the asteroid. 
        encoding: the encoding for the neural inputs.
        N (int): number of points. If a grid is passed should match that
        verbose (bool, optional): Print intermediate results. Defaults to False.
        noise (float): random noise added to point positions.
        sample_points (torch tensor): grid to sample the integral on
        h (float): grid spacing, only has to be passed if grid is passed.
        domain (torch.tensor): integration domain [3,2] , pass None for [-1,1]^3. Currently Not Implemented!

    Returns:
        Tensor: Computed potentials per point
    """
    if domain is not None:
        raise NotImplementedError(
            "Custom domain is not yet implemented for U_trap_opt.")

    # init result vector
    retval = torch.empty(len(target_points), 1,
                         device=os.environ["TORCH_DEVICE"])

    # Determine grid to compute on
    if sample_points is None:
        sample_points, h, N = compute_integration_grid(N, noise)
    else:
        if h is None:
            raise(ValueError("h has to be passed if sample points are passed."))

    # Evaluate Rho on the grid
    rho = _compute_model_output(model, encoding, sample_points)

    for i, target_point in enumerate(target_points):

        # Evaluate all points
        f_values = rho/torch.norm(target_point -
                                  sample_points, dim=1).view(-1, 1).detach()

        evaluations = f_values.reshape([N, N, N])  # map to z,y,x

        # area = h / 2 * (f0 + f2)
        int_x = h[0] / 2 * (evaluations[:, :, 0:-1] + evaluations[:, :, 1:])
        int_x = torch.sum(int_x, dim=2)
        int_y = h[1] / 2 * (int_x[:, 0:-1] + int_x[:, 1:])
        int_y = torch.sum(int_y, dim=1)
        int_z = h[2] / 2 * (int_y[0:-1] + int_y[1:])
        int_z = torch.sum(int_z, dim=0)

        retval[i] = int_z
    return -retval

# Low-discrepancy Montecarlo for the acceleration


def ACC_ld(target_points, model, encoding=direct_encoding(), N=3000, noise=0., domain=None):
    """Low discrepancy Monte Carlo evaluation of the potential from the modelled density

    Args:
        target_points (2-D array-like): a (N,3) array-like object containing the points.
        model (callable (a,b)->1): neural model for the asteroid. 
        encoding: the encoding for the neural inputs.
        N (int): number of points.
        noise (float): random noise added to point positions.
        domain (torch.tensor): integration domain [3,2] , pass None for [-1,1]^3. Currently Not Implemented!
    """
    if domain is not None:
        raise NotImplementedError(
            "Custom domain is not yet implemented for ACC_ld.")

    # init result vector
    retval = torch.empty(len(target_points), 3,
                         device=os.environ["TORCH_DEVICE"])

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

    # Evaluate Rho on the grid
    rho = _compute_model_output(model, encoding, sample_points)

    # the mc integral in the hypercube [-1,1]^3 (volume is 8) for each of the target points
    for i, target_point in enumerate(target_points):
        dr = torch.sub(target_point, sample_points)
        retval[i] = torch.sum(
            rho/torch.pow(torch.norm(dr, dim=1), 3).view(-1, 1) * dr, dim=0) / N
    return - 8 * retval


def ACC_trap(target_points, model, encoding=direct_encoding(), N=10000, verbose=False, noise=0., sample_points=None, h=None, domain=None):
    """Uses a 3D trapezoid rule for the evaluation of the integral in the potential from the modeled density

    Args:
        target_points (2-D array-like): a (N,3) array-like object containing the points.
        model (callable (a,b)->1): neural model for the asteroid. 
        encoding: the encoding for the neural inputs.
        N (int): number of points. If a grid is passed should match that
        verbose (bool, optional): Print intermediate results. Defaults to False.
        noise (float): random noise added to point positions.
        sample_points (torch tensor): grid to sample the integral on
        h (float): grid spacing, only has to be passed if grid is passed.
        domain (torch.tensor): integration domain [3,2] , pass None for [-1,1]^3 

    Returns:
        Tensor: Computed potentials per point
    """

    if domain is None:  # None might be passed as well
        domain = [[-1, 1], [-1, 1], [-1, 1]]

    # init result vector
    retval = torch.empty(len(target_points), 3,
                         device=os.environ["TORCH_DEVICE"])

    # Determine grid to compute on
    if sample_points is None:
        sample_points, h, N = compute_integration_grid(N, noise, domain)
    else:
        if h is None:
            raise(ValueError("h has to be passed if sample points are passed."))

    # Evaluate Rho on the grid
    rho = _compute_model_output(model, encoding, sample_points)

    for i, target_point in enumerate(target_points):

        # Evaluate all points
        distance = torch.sub(target_point, sample_points)
        f_values = (rho /
                    torch.pow(torch.norm(distance, dim=1), 3).view(-1, 1) * distance)

        evaluations = f_values.reshape([N, N, N, 3])  # map to z,y,x

        # area = h / 2 * (f0 + f2)
        int_x = h[0] / 2 * (evaluations[:, :, 0:-1, :] +
                            evaluations[:, :, 1:, :])
        int_x = torch.sum(int_x, dim=2)
        int_y = h[1] / 2 * (int_x[:, 0:-1, :] + int_x[:, 1:, :])
        int_y = torch.sum(int_y, dim=1)
        int_z = h[2] / 2 * (int_y[0:-1, :] + int_y[1:, :])
        int_z = torch.sum(int_z, dim=0)

        retval[i] = int_z
    return -retval


def compute_integration_grid(N, noise=0.0, domain=[[-1, 1], [-1, 1], [-1, 1]]):
    """Creates a grid which can be used for the trapezoid integration

    Args:
        N (int): Number of points to approximately  generate
        noise (float, optional): Amount of noise to add to points (can be used to sample nearby points). Defaults to 0.
        domain (torch.tensor): integration domain [3,2]

    Returns:
        torch tensor, float, int: sample points, grid h, nr of points
    """
    N = int(np.round(np.cbrt(N)))  # approximate subdivisions

    h = torch.zeros([3], device=os.environ["TORCH_DEVICE"])
    # Create grid and assemble evaluation points

    grid_1d_x = torch.linspace(
        domain[0][0], domain[0][1], N, device=os.environ["TORCH_DEVICE"])
    grid_1d_y = torch.linspace(
        domain[1][0], domain[1][1], N, device=os.environ["TORCH_DEVICE"])
    grid_1d_z = torch.linspace(
        domain[2][0], domain[2][1], N, device=os.environ["TORCH_DEVICE"])

    h[0] = (grid_1d_x[1] - grid_1d_x[0])
    h[1] = (grid_1d_y[1] - grid_1d_y[0])
    h[2] = (grid_1d_z[1] - grid_1d_z[0])

    x, y, z = torch.meshgrid(grid_1d_x, grid_1d_y, grid_1d_z)
    eval_points = torch.stack((x.flatten(), y.flatten(), z.flatten())).transpose(
        0, 1).to(os.environ["TORCH_DEVICE"])

    # We add some noise to the evaluated grid points to ensure the networks learns all
    if noise > 0:
        eval_points += torch.rand(N**3, 3,
                                  device=os.environ["TORCH_DEVICE"]) * noise

    return eval_points, h, N


def _check_model_encoding_compatibility(model, encoding):
    """ We check that the model is compatible with the encoding in terms of number of inputs

    Args:
        model (torch model): model to check
        encoding (encoding): encoding to use for model input

    Raises:
        ValueError: Raises error if model in features != encoding out dim 
    """
    if model.in_features != encoding.dim:
        print("encoding is incompatible with the model")
        raise ValueError


def _compute_model_output(model, encoding, sample_points):
    """Computes model output on the passed points using the passed encoding

    Args:
        model (torch model): neural network to eval
        encoding (encoding): encoding for network input (dim has to match)
        sample_points (torch tensor): points to sample at

    Returns:
        torch tensor: computed values
    """
    # check dimensions match
    _check_model_encoding_compatibility(model, encoding)

    # 1 - compute the inputs to the ANN encoding the sampled points
    nn_inputs = encoding(sample_points)

    # 2 - check if any values were NaN
    if torch.any(nn_inputs != nn_inputs):
        warnings.warn("The network generated NaN outputs!")
        nn_inputs[nn_inputs != nn_inputs] = 0.0  # set Nans to 0

    # 3 - compute the predicted density at the points
    return model(nn_inputs)
