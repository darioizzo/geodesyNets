import torch
import os


def ACC_L_differential(target_points, mascon_points, mascon_masses_u, mascon_masses_nu):
    """Computes the difference in acceleration between the uniform and non-uniform mascon model

    Args:
        target_points (2-D array-like): an (N, 3) array-like object containing the coordinates of the points where the acceleration should be computed.
        mascon_points (2-D array-like): an (N, 3) array-like object containing the points that belong to the mascon
        mascon_masses_u (1-D array-like): a (N,) array-like object containing the values for the uniform mascon masses. Can also be a scalar containing the mass value for all points.
        mascon_masses_nu (1-D array-like): a (N,) array-like object containing the values for the non-uniform mascon masses. Can also be a scalar containing the mass value for all points.
    """
    labels_u = ACC_L(target_points, mascon_points, mascon_masses_u)
    labels_nu = ACC_L(target_points, mascon_points, mascon_masses_nu)
    labels = labels_nu - labels_u
    return labels


def U_L(target_points, mascon_points, mascon_masses=None):
    """
    Computes the gravity potential (G=1) created by a mascon in the target points. (to be used as Label in the training)

    Args:
        target_points (2-D array-like): an (N, 3) array-like object containing the coordinates of the points where the potential is seeked.
        mascon_points (2-D array-like): an (N, 3) array-like object containing the points that belong to the mascon
        mascon_masses (1-D array-like): a (N,) array-like object containing the values for the mascon masses. Can also be a scalar containing the mass value for all points.

    Returns:
        1-D array-like: a (N, 1) torch tensor containing the gravity potential (G=1) at the target points
    """

    if mascon_masses is None:
        mascon_masses = 1./len(mascon_points)
    retval = torch.empty(len(target_points), 1,
                         device=os.environ["TORCH_DEVICE"])
    for i, target_point in enumerate(target_points):
        retval[i] = torch.sum(
            mascon_masses/torch.norm(torch.sub(mascon_points, target_point), dim=1))
    return - retval


def ACC_L(target_points, mascon_points, mascon_masses=None):
    """
    Computes the acceleration due to the mascon at the target points. (to be used as Label in the training)

    Args:
        target_points (2-D array-like): an (N, 3) array-like object containing the coordinates of the points where the  acceleration should be computed.
        mascon_points (2-D array-like): an (N, 3) array-like object containing the points that belong to the mascon
        mascon_masses (1-D array-like): a (N,) array-like object containing the values for the mascon masses. Can also be a scalar containing the mass value for all points.

    Returns:
        1-D array-like: a (N, 3) torch tensor containing the acceleration (G=1) at the target points
    """
    if mascon_masses is None:
        mm = torch.tensor([1./len(mascon_points)] * len(mascon_points),
                          device=os.environ["TORCH_DEVICE"]).view(-1, 1)
    elif type(mascon_masses) is int:
        mm = torch.tensor([mascon_masses] * len(mascon_points),
                          device=os.environ["TORCH_DEVICE"]).view(-1, 1)
    else:
        mm = mascon_masses.view(-1, 1)
    retval = torch.empty(len(target_points),  3,
                         device=os.environ["TORCH_DEVICE"])
    for i, target_point in enumerate(target_points):
        dr = torch.sub(mascon_points, target_point)
        retval[i] = torch.sum(
            mm/torch.pow(torch.norm(dr, dim=1), 3).view(-1, 1) * dr, dim=0)
    return retval
