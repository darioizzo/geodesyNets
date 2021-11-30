import torch
import warnings
from ._encodings import direct_encoding

import os

import numpy as np

from .torchquad.integration.base_integrator import BaseIntegrator
from .torchquad.integration.integration_grid import IntegrationGrid
from .torchquad.integration.utils import _setup_integration_domain

def tq_integrate(target_points, model, encoding=direct_encoding(), N=10000, grid=None):
    """Uses a 3D boole rule for the evaluation of the integral in the potential from the modeled density

        Args:
            target_points (2-D array-like): a (N,3) array-like object containing the points.
            model (callable (a,b)->1): neural model for the asteroid. 
            encoding: the encoding for the neural inputs.
            N (int): number of points. If a grid is passed should match that
            integrator (str): which integrator to use. Defaults to Boole.
            domain (torch.tensor): integration domain [3,2] , pass None for [-1,1]^3.

        Returns:
            Tensor: Computed potentials per point
        """
    integ = TQ_Integrator()
    return integ.integrate(target_points, model, encoding=encoding, N=N, grid=grid)


class TQ_Integrator(BaseIntegrator):
    """Boole's rule in torch. See https://en.wikipedia.org/wiki/Newton%E2%80%93Cotes_formulas#Closed_Newton%E2%80%93Cotes_formulas ."""

    def __init__(self):
        super().__init__()

    def _adjust_N(self, dim, N):
        """Adjusts the total number of points to a valid number, i.e. N satisfies N^(1/dim) - 1 % 4 == 0.

        Args:
            dim (int): Dimensionality of the integration domain.
            N (int): Total number of sample points to use for the integration.

        Returns:
            int: An N satisfying N^(1/dim) - 1 % 4 == 0.
        """
        n_per_dim = int(N ** (1.0 / dim) + 1e-8)

        # Boole's rule requires N per dim >=5 and N = 1 + 4n,
        # where n is a positive integer, for correctness.
        if n_per_dim < 5:
            warnings.warn(
                "N per dimension cannot be lower than 5. "
                "N per dim will now be changed to 5."
            )
            N = 5 ** dim
        elif (n_per_dim - 1) % 4 != 0:
            new_n_per_dim = n_per_dim - ((n_per_dim - 1) % 4)
            warnings.warn(
                "N per dimension must be N = 1 + 4n with n a positive integer due to necessary subdivisions. "
                "N per dim will now be changed to the next lower N satisfying this, i.e. "
                f"{n_per_dim} -> {new_n_per_dim}."
            )
            N = (new_n_per_dim) ** (dim)
        return N

    def integrate(self,target_points, model, encoding=direct_encoding(), N=10000, grid=None):
        """Uses a 3D boole rule for the evaluation of the integral in the potential from the modeled density

        Args:
            target_points (2-D array-like): a (N,3) array-like object containing the points.
            model (callable (a,b)->1): neural model for the asteroid. 
            encoding: the encoding for the neural inputs.
            N (int): number of points. If a grid is passed should match that
            integrator (str): which integrator to use. Defaults to Boole.
            domain (torch.tensor): integration domain [3,2] , pass None for [-1,1]^3.

        Returns:
            Tensor: Computed potentials per point
        """

        domain = [[-1, 1], [-1, 1], [-1, 1]]

        self._integration_domain = _setup_integration_domain(3, domain)
        self._check_inputs(dim=3, N=N, integration_domain=self._integration_domain)
        N = self._adjust_N(dim=3, N=N)

        # init result vector
        retval = torch.empty(len(target_points), 3,device=os.environ["TORCH_DEVICE"])

        self._dim = 3

        if grid == None:
            self._grid = IntegrationGrid(N, self._integration_domain)
        else:
            self._grid = grid

        # Evaluate Rho on the grid
        rho = self._compute_model_output(model, encoding, self._grid.points)

        for i, target_point in enumerate(target_points):

            # Evaluate all points
            distance = torch.sub(target_point, self._grid.points)
            f_values = (rho /
                        torch.pow(torch.norm(distance, dim=1), 3).view(-1, 1) * distance)
            
            
            f_values = f_values.reshape([self._grid._N]*3+[3])

            # We collapse dimension by dimension
            for cur_dim in range(3):
                f_values = (
                    self._grid.h[cur_dim]
                    / 22.5
                    * (
                        7 * f_values[..., 0:-4,:][..., ::4,:]
                        + 32 * f_values[..., 1:-3,:][..., ::4,:]
                        + 12 * f_values[..., 2:-2,:][..., ::4,:]
                        + 32 * f_values[..., 3:-1,:][..., ::4,:]
                        + 7 * f_values[..., 4:,:][..., ::4,:]
                    )
                )
                f_values = torch.sum(f_values, dim=3 - cur_dim - 1)
            retval[i]  = f_values
        return -retval

    def _check_model_encoding_compatibility(self,model, encoding):
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


    def _compute_model_output(self,model, encoding, sample_points):
        """Computes model output on the passed points using the passed encoding

        Args:
            model (torch model): neural network to eval
            encoding (encoding): encoding for network input (dim has to match)
            sample_points (torch tensor): points to sample at

        Returns:
            torch tensor: computed values
        """
        # check dimensions match
        self._check_model_encoding_compatibility(model, encoding)

        # 1 - compute the inputs to the ANN encoding the sampled points
        nn_inputs = encoding(sample_points)

        # 2 - check if any values were NaN
        if torch.any(nn_inputs != nn_inputs):
            warnings.warn("The network generated NaN outputs!")
            nn_inputs[nn_inputs != nn_inputs] = 0.0  # set Nans to 0

        # 3 - compute the predicted density at the points
        return model(nn_inputs)


def ACC_trap(target_points, model, encoding=direct_encoding(), N=10000, verbose=False, noise=1e-5, sample_points=None, h=None, domain=None):
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
