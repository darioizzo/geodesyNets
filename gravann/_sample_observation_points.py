import torch
import matplotlib.pyplot as plt
import numpy as np
import os


def get_target_point_sampler(N, method="default"):
    """Get a function to sample N target points from. Points may differ each
    call depending on selected method. See specific implementations for details.

    Args:
        N (int): Number of points to get each call
        method (str, optional): Utilized method. Currently supports random points
                                (default) or a spherical grid (will not change each 
                                call). Defaults to "default".

    Returns:
        lambda: function to call to get sampled target points
    """
    if method == "default":
        return lambda: _sample_default(N)
    elif method == "spherical_grid":
        points = _get_spherical_grid(N)
        return lambda: points


def _get_spherical_grid(N, radius=1.73205):
    """Projects a 2D grid onto a sphere with an offset at the poles to avoid singularities.
    Will only return square numbers of points.      

    Args:
        N (int): Approximate number of points to create (lower square nr below will be selected in practice)
        radius (float, optional): [description]. Defaults to 1.73205 which is approximately corner of unit cube..

    Returns:
        [torch tensor]: Points on the sphere.
    """
    N = int(np.round(np.sqrt(N)))  # 2d grid
    offset = np.pi / (N+2)  # Use an offset to avoid singularities at poles
    grid_1d = torch.linspace(
        offset, np.pi-offset, N, device=os.environ["TORCH_DEVICE"])
    phi, theta = torch.meshgrid(grid_1d, grid_1d)
    x = radius * torch.sin(phi) * torch.cos(2*theta)
    y = radius * torch.sin(phi) * torch.sin(2*theta)
    z = radius * torch.cos(phi)
    points = torch.stack((x.flatten(), y.flatten(), z.flatten())).transpose(
        0, 1).to(os.environ["TORCH_DEVICE"])
    return points


def _limit_to_domain(points, domain=[[-1, 1], [-1, 1], [-1, 1]]):
    """Throws away all passed points that were inside the passed domain. Domain has to be convex.

    Args:
        points (torch tensor): Points to inspect.
        domain (list, optional): Domain to use. Defaults to [[-1, 1], [-1, 1], [-1, 1]].

    Returns:
        Torch tensor: (Non-proper) subset of the passed points.
    """
    a = torch.logical_or((points[:, 0] < domain[0][0]),
                         (points[:, 0] > domain[0][1]))
    b = torch.logical_or((points[:, 1] < domain[1][0]),
                         (points[:, 1] > domain[1][1]))
    c = torch.logical_or((points[:, 2] < domain[2][0]),
                         (points[:, 2] > domain[2][1]))
    d = torch.logical_or(torch.logical_or(a, b), c)
    return points[d]


def _sample_default(N, scale=1.1):
    """Generates N uniform random samples from a cube with passed scale. All points outside unit cube.

    Args:
        N (int): Nr of points to create.
        scale (float, optional): Scale of the domain for the points (unitcube will be discarded so has to be > 1). Defaults to 1.1.

    Returns:
        Torch tensor: Sampled points
    """
    # Approximation of percentage points in Unitsphere
    approx = (1.0 / scale)**3

    # Sample twice the expected number of points necessary to achieve N
    approx_necessary_samples = int(2 * N * (1.0 / (1.0 - approx)))
    points = (torch.rand(approx_necessary_samples, 3)*2 - 1)*scale

    # Discard points inside unitcube
    points = _limit_to_domain(points)

    # Take first N points (N.B. that in super unlikely event of
    # less than N points this will not crash. I tried. :))
    return points[:N]
