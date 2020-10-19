"""
This module helps in the design and analysis of Artificial Neural Networks to represent the gravity field of celestial objects.
It was developed by the Advanced Conpcets team in the context of the project "ANNs for geodesy".
"""
import torch
import warnings
import os

# Importing encodings for the spacial asteroid dimensions
from ._encodings import directional_encoding, positional_encoding, direct_encoding, spherical_coordinates

# Importing the losses
from ._losses import normalized_loss, mse_loss

# Importing the method to integrate the density rho(x,y,z) output of an ANN in the unit cube
from ._integration import U_Pmc, U_Pld, U_trap_opt

# Importing alpha shape methods
from ._hulls import alpha_shape

# Importing the plots
from ._plots import plot_mascon, plot_model_grid, plot_model_rejection
from ._plots import plot_mesh, plot_model_mesh, plot_point_cloud_mesh, plot_points

# Importing methods to sample points around asteroid
from ._sample_observation_points import get_target_point_sampler

# Importing the mesh_conversion methods
from ._mesh_conversion import create_mesh_from_cloud, create_mesh_from_model

from ._utils import max_min_distance

# Will track the main device to use
os.environ["TORCH_DEVICE"] = 'cpu'

# Miscellaneous functions loaded into the main namespace
def enableCUDA(device=0):
    """This function will set the default device to CUDA if possible. Call before declaring any variables!
    """
    if torch.cuda.is_available():
        os.environ["TORCH_DEVICE"] = "cuda:" + str(device)
        print('Available devices ', torch.cuda.device_count())
        print('__pyTorch VERSION:', torch.__version__)
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        print('__Number CUDA Devices:', torch.cuda.device_count())
        print('Active CUDA Device: GPU', torch.cuda.current_device())
        print("Setting default tensor type to Float32")
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        warnings.warn(
            "Error enabling CUDA. cuda.is_available() returned False. CPU will be used.")


def U_L(target_points, points, masses=None):
    """
    Computes the gravity potential created by a mascon in the target points.

    Args:
        target_points (2-D array-like): an (N, 3) array-like object containing the coordinates of the points where the potential is seeked.
        points (2-D array-like): an (N, 3) array-like object containing the coordinates of the points
        masses (1-D array-like): a (N,) array-like object containing the values for the point masses. Can also be a scalar containing the mass value for all points.
    """

    if masses is None:
        masses = 1./len(points)
    retval = torch.empty(len(target_points), 1,
                         device=os.environ["TORCH_DEVICE"])
    for i, target_point in enumerate(target_points):
        retval[i] = torch.sum(
            masses/torch.norm(torch.sub(points, target_point), dim=1))
    return - retval
