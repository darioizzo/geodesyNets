import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pk
import pyvista as pv
import warnings
from scipy.spatial import KDTree

from ._utils import unpack_triangle_mesh, get_asteroid_bounding_box
from ._hulls import is_outside_torch, is_outside

# There is no torch.pi so we define it here
torch.pi = torch.acos(torch.zeros(1)).item() * 2  # which is 3.1415927410125732


def get_target_point_sampler(N, method="cubical", bounds=[1.1, 1.2], limit_shape_to_asteroid=None, replace=True):
    """Get a function to sample N target points from. Points may differ each
    call depending on selected method. See specific implementations for details.

    Args:
        N (int): Number of points to get each call
        radius_bounds (list): Defaults to [1.1, 1.2]. Specifies the sampling radius.
        method (str, optional): Utilized method. Currently supports random points from some volume
                                (cubical, spherical) or a spherical grid (will not change each
                                call). Defaults to "cubical".
        limit_shape_to_asteroid(str, optional): Path to a *.pk file specifies an asteroid shape to exclude from samples
                                                or use for altitude sampling
        replace (bool, optional): Only altitude. If points are allowed to be sampled twice in the same batch or not
                                  (for false maximum sample points = #triangles in mesh)

    Returns:
        lambda: function to call to get sampled target points
    """
    if limit_shape_to_asteroid == None:
        # Create point sampler function
        if method == "cubical":
            return lambda: _sample_cubical(N, bounds)
        elif method == "spherical":
            return lambda: _sample_spherical(N, bounds)
        elif method == "spherical_grid":
            points = _get_spherical_grid(N)
            return points
        else:
            raise ValueError("Method combination not available.")
    # Create domain limiter if passed
    else:
        if method == "altitude":
            return _get_altitude_sampler(N, bounds[0], limit_shape_to_asteroid, replace=replace)
        elif method == "radial_projection":
            return lambda: _get_radial_projection_sampler(N, altitude=bounds, limit_shape_to_asteroid=limit_shape_to_asteroid)
        else:
            return _get_asteroid_limited_sampler(
                N, method, bounds, limit_shape_to_asteroid)


def _get_radial_projection_sampler(steps, altitude, limit_shape_to_asteroid, debug_plot=False):
    """ Sample points by radially projection outward from asteroid center
    as in Wittick & Russel - Mixed-model gravity representations for small celestial bodies

    Args:
        steps (int): discretization steps for the sampling
        altitude (float): altitude to project by
        limit_shape_to_asteroid (str): path of to asteroid mesh
        debug_plot (bool): Can be used to plot created points.

    Raises:
        ValueError: If too many points are requested

    Returns:
        [type]: [description]
    """
    #
    # Load asteroid vertices
    with open(limit_shape_to_asteroid, "rb") as file:
        mesh_vertices, mesh_triangles = pk.load(file)

    # Get bounding box and compute central point from that.
    bb = get_asteroid_bounding_box(limit_shape_to_asteroid)
    center = [0.5 * (bb[0][0] + bb[0][1]),
              0.5 * (bb[1][0] + bb[1][1]),
              0.5 * (bb[2][0] + bb[2][1])]

    # Pick radius as maximal extension of the asteroid in one dim.
    # Note this may cause problems for asteroids that extend beyond the sphere with that radius.
    # Therefore we later discard points which accidentally end up in the asteroid.
    r_a = np.maximum(np.maximum(bb[0][1]-bb[0][0], bb[1][1]-bb[1][0]),
                     bb[2][1]-bb[2][0]) / 2.0

    N = len(mesh_vertices)
    points = np.zeros([steps*N, 3])

    # Sample different altitudes
    for idx, grid_point in enumerate(np.linspace(altitude[0]+1e-4, altitude[1], steps)):
        # Sample points by sampling random altitude between altitude[0] and altitude[1] times r_a
        current_altitude = grid_point * r_a
        points[idx*N:(idx+1)*N, :] = mesh_vertices + \
            (np.asarray(mesh_vertices) - center) * current_altitude

    if debug_plot:
        mesh_faces = [[3, t[0], t[1], t[2]] for t in mesh_triangles]
        # Create PV Polydata
        mesh = pv.PolyData(np.asarray(mesh_vertices), np.asarray(mesh_faces))
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color='brown')
        plotter.add_mesh(points, color='red')
        plotter.show_grid()
        plotter.show()

    return torch.tensor(points)


def _get_altitude_sampler(N, altitude, limit_shape_to_asteroid, plot_normals=False, discard_points_inside=True, replace=True):
    """This creates a sampler that samples from the triangle centers of the passed mesh + their normal

    Args:
        N (int): number of points to get per batch
        altitude (float): altitude above the faces
        limit_shape_to_asteroid (str): path of to asteroid mesh
        plot_normals (bool, optional): Display normals and created points for debugging. Defaults to False.
        discard_points_inside (bool, optional): Will discard all points that lie inside the asteroid (can happen for nonconvex ones). Defaults to True.
        replace (bool, optional): If points are allowed to be sampled twice in the same batch or not
                                  (for false maximum sample points = #triangles in mesh)

    Returns:
        func: sampler function
    """
    # print("Sampling ", limit_shape_to_asteroid, " at altitude=", altitude)

    # Load asteroid triangles
    with open(limit_shape_to_asteroid, "rb") as file:
        mesh_vertices, mesh_triangles = pk.load(file)
        mesh_faces = [[3, t[0], t[1], t[2]] for t in mesh_triangles]

    # Create PV Polydata
    mesh = pv.PolyData(np.asarray(mesh_vertices), np.asarray(mesh_faces))
    # mesh.plot(show_scalar_bar=False)
    mesh.compute_normals(cell_normals=True, point_normals=False,
                         inplace=True, consistent_normals=False,
                         non_manifold_traversal=False)

    centers = mesh.cell_centers().points
    points_at_altitude = centers + altitude * mesh['Normals']

    if plot_normals:
        # mesh.plot()
        pv.set_plot_theme("document")
        plotter = pv.Plotter()
        point_cloud = pv.PolyData(centers)
        point_cloud["vectors"] = mesh['Normals']
        arrows = point_cloud.glyph(
            orient='vectors', scale=True, factor=0.1)
        plotter.add_mesh(mesh, color=[0.2, 0.2, 0.2], lighting=True,
                         smooth_shading=False, show_edges=True, edge_color=[0.25, 0.25, 0.25])
        plotter.add_mesh(arrows, opacity=1.0, color='lightblue')
        plotter.add_mesh(points_at_altitude, style="points",
                         color='red', point_size=4, render_points_as_spheres=True)
        plotter.show_grid()
        # plotter.show()
        plotter.show(screenshot="figures/sampler.png", window_size=[1200, 800])

    # Discard points that are too close (or too far)
    eps = 1e-4  # maximum altitude error
    kd_tree = KDTree(centers)
    distances, _ = kd_tree.query(points_at_altitude, k=1)
    distance_correct = np.abs(altitude-distances) < eps
    print("Discarding " + str(len(distance_correct) - np.sum(distance_correct)) + " of " + str(len(distance_correct)) +
          " points in altitude sampler which did not meet requested altitude.")
    points_at_altitude = points_at_altitude[distance_correct]

    if discard_points_inside:
        # print("Discarding points inside asteroid. Prev len=",
        #   len(points_at_altitude))
        triangles = unpack_triangle_mesh(mesh_vertices, mesh_triangles)
        points_at_altitude = points_at_altitude[is_outside_torch(
            torch.tensor(points_at_altitude).float().detach(), triangles).detach().cpu().numpy()]
        del triangles
        # print("Now len=", len(points_at_altitude))

    return lambda: torch.tensor(points_at_altitude[np.random.choice(
        points_at_altitude.shape[0], N, replace=replace), :])


def _get_asteroid_limited_sampler(N, method="cubical", bounds=[1.1, 1.2], limit_shape_to_asteroid=None, sample_step_size=32):
    """Get a function to sample N target points from. Points may differ each
    call depending on selected method. See specific implementations for details.

    Args:
        N (int): Number of points to get each call
        bounds (list): Defaults to [1.1, 1.2]. Specifies the sampling radius. (First entry used as altitude for altitude sampling)
        method (str, optional): Utilized method. Currently supports random points from some volume
                                (cubical, spherical) or a spherical grid (will not change each
                                call). Defaults to "cubical".
        limit_shape_to_asteroid(str, optional): Path to a *.pk file specifies an asteroid shape to exclude from samples
        sample_step_size (int, optional): How many samples are drawn each try to add until N reached

    Returns:
        lambda: function to call to get sampled target points
    """
    if method == "spherical_grid":
        raise NotImplementedError(
            "Sorry. Spherical grid is not supported with asteroid shape as the grid is the same on each call.")

    # Load asteroid triangles
    with open(limit_shape_to_asteroid, "rb") as file:
        mesh_vertices, mesh_triangles = pk.load(file)
        triangles = unpack_triangle_mesh(mesh_vertices, mesh_triangles)

    # Create a sampler to get some points
    if method == "cubical":
        def sampler(): return _sample_cubical(sample_step_size, bounds)
    elif method == "spherical":
        def sampler(): return _sample_spherical(sample_step_size, bounds)

    # Create a new sampler inside the sampler so to speak
    return lambda: _get_N_points_outside_asteroid(N, sampler, triangles, sample_step_size)


def _get_N_points_outside_asteroid(N, sampler, triangles, sample_step_size):
    """Sample points until N outside asteroid reached with given sampler and triangles

    Args:
        N (int): target # of points
        sampler (func): sampler to call
        triangles (torch tensor): triangles of the asteroid as (v0,v1,v2)
        sample_step_size (int): how many points the sampler will spit out each call

    Returns:
        torch tensor: sampled points
    """
    # We allocate a few more just to avoid having to check, will discard in return
    points = torch.zeros([N+sample_step_size, 3],
                         device=os.environ["TORCH_DEVICE"])
    found_points = 0

    # Sample points till we sufficient amount
    while found_points < N:

        # Get some points
        candidates = sampler()
        candidates_outside = candidates[is_outside_torch(
            candidates, triangles)]

        # Add those that were outside to our collection
        new_points = len(candidates_outside)
        if new_points > 0:
            points[found_points:found_points +
                   new_points, :] = candidates_outside
            found_points += new_points

    return points[:N]


def _get_spherical_grid(N, radius=1.73205):
    """Projects a 2D grid onto a sphere with an offset at the poles to avoid singularities.
    Will only return square numbers of points.

    Args:
        N (int): Approximate number of points to create (lower square nr below will be selected in practice)
        radius (float, optional): [description]. Defaults to 1.73205 which is approximately corner of unit cube.

    Returns:
        [torch tensor]: Points on the sphere.
    """
    N = int(np.round(np.sqrt(N)))  # 2d grid
    offset = torch.pi / (N+2)  # Use an offset to avoid singularities at poles
    grid_1d = torch.linspace(
        offset, torch.pi-offset, N, device=os.environ["TORCH_DEVICE"])
    phi, theta = torch.meshgrid(grid_1d, grid_1d)
    x = radius * torch.sin(phi) * torch.cos(2*theta)
    y = radius * torch.sin(phi) * torch.sin(2*theta)
    z = radius * torch.cos(phi)
    points = torch.stack((x.flatten(), y.flatten(), z.flatten())).transpose(
        0, 1).to(os.environ["TORCH_DEVICE"])
    return points


def _limit_to_domain(points, domain=[[-1, 1], [-1, 1], [-1, 1]]):
    """Throws away all passed points that were inside the passed domain. Domain has to be cuboid.

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


def _sample_spherical(N, radius_bounds=[1.1, 1.2]):
    """Generates N uniform random samples inside a sphere with specified radius bounds.

    Args:
        N (int): Number of points to create
        radius_bounds (float, optional): [description]. Defaults to [1.1, 1.2] which will create points also inside the unit cube.

    Returns:
        Torch tensor: Sampled points
    """
    theta = 2.0 * torch.pi * torch.rand(N, 1,
                                        device=os.environ["TORCH_DEVICE"])

    # The acos here allows us to sample uniformly on the sphere
    phi = torch.acos(1.0 - 2.0 * torch.rand(N, 1,
                                            device=os.environ["TORCH_DEVICE"]))

    minimal_radius_scale = radius_bounds[0] / radius_bounds[1]
    # Create uniform between
    uni = minimal_radius_scale + \
        (1.0 - minimal_radius_scale) * \
        torch.rand(N, 1, device=os.environ["TORCH_DEVICE"])
    r = radius_bounds[1] * torch.pow(uni, 1/3)

    x = r * torch.sin(phi) * torch.cos(theta)
    y = r * torch.sin(phi) * torch.sin(theta)
    z = r * torch.cos(phi)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten())).transpose(
        0, 1).to(os.environ["TORCH_DEVICE"])

    if os.environ["TORCH_DEVICE"] == "cpu":
        return points
    else:
        return points.float()


def _sample_cubical(N, scale_bounds=[1.1, 1.2]):
    """Generates N uniform random samples from a cube with passed scale. All points outside unit cube.

    Args:
        N (int): Nr of points to create.
        scale_bounds (float, optional): Scales of the domain for the points. Defaults to [1.1, 1.2].

    Returns:
        Torch tensor: Sampled points
    """
    # Approximation of percentage points in Unitsphere
    approx = (scale_bounds[0] / scale_bounds[1])**3

    # Sample twice the expected number of points necessary to achieve N
    approx_necessary_samples = int(2 * N * (1.0 / (1.0 - approx)))
    points = (torch.rand(approx_necessary_samples, 3,
                         device=os.environ["TORCH_DEVICE"])*2 - 1)*scale_bounds[1]

    # Discard points inside unitcube
    domain = scale_bounds[0]*torch.tensor(
        [[-1, 1], [-1, 1], [-1, 1]], device=os.environ["TORCH_DEVICE"])
    points = _limit_to_domain(points, domain)

    # Take first N points (N.B. that in super unlikely event of
    # less than N points this will not crash. I tried. :))
    return points[:N]
