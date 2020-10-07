"""
This module helps in the design and analysis of Artificial Neural Networks to represent the gravity field of celestial objects.
It was developed by the Advanced Conpcets team in the context of the project "ANNs for geodesy".
"""
import torch

# Importing encodings for the spacial asteroid dimensions
from ._encodings import directional_encoding, positional_encoding, direct_encoding, spherical_coordinates

# Importing the losses
from ._losses import normalized_loss, mse_loss

# Importing the method to integrate the density rho(x,y,z) output of an ANN in the unit cube
from ._integration import U_Pmc, U_Pld

# Importing the plots
from ._plots import plot_mascon, plot_model_grid, plot_model_rejection


# Miscellaneous functions loaded into the main namespace
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
    retval = torch.empty(len(target_points), 1)
    for i, target_point in enumerate(target_points):
        retval[i] = torch.sum(
            masses/torch.norm(torch.sub(points, target_point), dim=1))
    return - retval
