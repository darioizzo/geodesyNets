import numpy as np
import torch
import os

import pyvista as pv

# This function computes the distance of some points `target_points`to the `cloud_points` per target point as
# min(distance(target_point,cloud_points))


def _point_cloud_distance(target_points, cloud_points):
    """ This function computes the distance of some points `target_points`to the `cloud_points` per target point as min(distance(target_point,cloud_points)) 

    Args:
        target_points (numpy arr): Points to compute distance for
        cloud_points (numpy arr): Point cloud

    Returns:
        np array: Per point minimal distance to cloud points
    """
    distances = np.zeros(len(target_points))
    # per point compute distance to the cloud
    for i, point in enumerate(target_points):
        distances[i] = np.min(np.linalg.norm(point-cloud_points, axis=1))
    return distances


def _point_cloud_topk_distance(target_points, cloud_points, k=5):
    """ This function computes the distance of some points `target_points`to the `cloud_points` per target point as mean distance of k closest cloud points

    Args:
        target_points (numpy arr): Points to compute distance for
        cloud_points (numpy arr): Point cloud
        k (int): number of neighbors to consider

    Returns:
        np array: Per point mean distance to 5 closest cloud points
    """
    distances = np.zeros(len(target_points))
    # per point compute distance to the cloud
    for i, point in enumerate(target_points):
        dist = np.linalg.norm(point-cloud_points, axis=1)
        # get 5 smallest elements and sum them
        idx = np.argpartition(dist, k)
        distances[i] = np.sum(dist[idx[:k]]) / k
    return distances


def create_mesh_from_cloud(cloud_points, cube_scale=1, subdivisions=6, stepsize=0.005,
                           target_point=[0, 0, 0], distance_threshold=0.125, adaptive_step=True,
                           verbose=False, plot_each_it=10, max_iter=200, use_top_k=1):
    """Generates a mesh from a point cloud

    Args:
        cloud_points (numpy array): point_cloud points (N,3)
        cube_scale (int): scale dimension of the cube 1 leads to [-1,1]^3
        subdivisions (int): mesh granularity (pick 4 to 7 or so)
        stepsize (float): stepsize each vertix takes per iteration, or initial one if adaptive stepsize is used
        target_point (tuple): the point vertices "fly" towards , default [0,0,0]
        distance_threshold (float): distance cutoff (0.125 good start for 5point, 0.05 for 1point)
        adaptive_step (bool): Use an adptive stepsize? Much faster
        verbose (bool): verbose out put for debugging
        plot_each_it (int): how often to plot
        max_iter (int): max iteration termination criterium
        use_top_k (int): the number of nearest neighbours to be used for distance.

    Returns:
        [pyvista mesh]: Mesh of the cloud
    """
    # Initialize cube mesh
    cube = pv.Cube(x_length=2*cube_scale, y_length=2*cube_scale,
                   z_length=2*cube_scale).triangulate()
    cube = cube.subdivide(subdivisions)
    N = len(cube.points)

    # Initialize per vertex normalized target direction (in which direction the vertex will travel)
    target_direction = target_point - cube.points
    for i in range(N):
        target_direction[i] = target_direction[i] / \
            np.linalg.norm(target_direction[i])

    # If adaptive stepsize, start each vertex with default stepsize
    if adaptive_step:
        stepsize = np.ones([N, 3]) * stepsize

    if verbose:
        print(cube)

    # defines the still moving points
    point_to_compute = np.asarray([True] * N)
    it = 0

    while any(point_to_compute) and it < max_iter:
        if it % plot_each_it == 0 and plot_each_it > 0:
            print(
                f"Iteration {it} - {sum(point_to_compute)} Travelling Vertices")

        # Get remaining points
        remaining_points = cube.points[point_to_compute]
        if verbose:
            print("remaining_points", remaining_points)

        # Compute new position of each point
        new_points = remaining_points + stepsize * \
            target_direction[point_to_compute]
        if verbose:
            print("new_points", new_points)

        # Compute values at new positions
        if use_top_k > 1:
            cloud_distances = _point_cloud_topk_distance(
                new_points, cloud_points, use_top_k)
        else:
            cloud_distances = _point_cloud_distance(new_points, cloud_points)
        if verbose:
            print("cloud_distances", cloud_distances)

        # Where value bigger than threshold stop changing this point
        point_to_compute[point_to_compute] = cloud_distances > distance_threshold
        if verbose:
            print("point_to_compute", point_to_compute)

        # Where value smaller than threshold change vertex to this position
        cube.points[point_to_compute] = new_points[cloud_distances >
                                                   distance_threshold]
        if verbose:
            print("cube.points", cube.points)

        if adaptive_step:
            dst = cloud_distances[cloud_distances > distance_threshold]
            if use_top_k > 1:
                stepsize = 0.00001 + 0.05 * np.stack([dst, dst, dst], axis=1)
            else:
                stepsize = 0.00001 + 0.1 * np.stack([dst, dst, dst], axis=1)

        if it % plot_each_it == 0 and plot_each_it > 0:
            p = pv.Plotter(window_size=[200, 100])
            p.add_mesh(cube, color="grey", show_edges=False)
            p.show()

        it += 1

    return cube


def create_mesh_from_model(model, encoding, cube_scale=1.5, subdivisions=6,  # mesh subdivs
                           stepsize=0.001, target_point=[0, 0, 0], rho_threshold=1.5e-2, verbose=False,
                           plot_each_it=100, max_iter=4000):
    """Generates a mesh from a model

    Args:
        model (Torch Model): Model to use 
        encoding (Encoding function): The function used to encode points for the model
        cube_scale (int): scale dimension of the cube 1 leads to [-1,1]^3
        subdivisions (int): mesh granularity (pick 4 to 7 or so)
        stepsize (float): stepsize each vertix takes per iteration, or initial one if adaptive stepsize is used
        target_point (tuple): the point vertices "fly" towards , default [0,0,0]
        rho_threshold (float): rho cutoff where points are considered to be inside. Defaults to 1.5e-2
        verbose (bool): verbose out put for debugging
        plot_each_it (int): how often to plot
        max_iter (int): max iteration termination criterium
    Returns:
        [pyvista mesh]: Mesh of the cloud
    """
    # Initialize cube mesh
    cube = pv.Cube(x_length=2*cube_scale, y_length=2*cube_scale,
                   z_length=2*cube_scale).triangulate()
    cube = cube.subdivide(subdivisions)
    N = len(cube.points)

    # Initialize per vertex normalized target direction (in which direction the vertex will travel)
    target_direction = target_point - cube.points
    for i in range(N):
        target_direction[i] = target_direction[i] / \
            np.linalg.norm(target_direction[i])

    if verbose:
        print(cube)

    # Will define which vertices are still traveling
    point_to_compute = np.asarray([True] * N)
    it = 0

    while any(point_to_compute) and it < max_iter:
        if it % plot_each_it == 0 and plot_each_it > 0:
            print(
                f"Iteration {it} - {sum(point_to_compute)} Travelling Vertices")

        # Get remaining points
        remaining_points = cube.points[point_to_compute]
        if verbose:
            print("remaining_points", remaining_points)

        # Compute new position of each point
        new_points = remaining_points + stepsize * \
            target_direction[point_to_compute]
        if verbose:
            print("new_points", new_points)

        # Compute values at new positions
        new_points = torch.tensor(
            new_points, dtype=torch.float, device=os.environ["TORCH_DEVICE"])
        nn_inputs = encoding(new_points)
        rho = model(nn_inputs).squeeze().detach().cpu().numpy()
        if verbose:
            print("rho", rho)

        # Where value bigger than threshold stop changing this point
        point_to_compute[point_to_compute] = rho < rho_threshold
        if verbose:
            print("point_to_compute", point_to_compute)

        # Where value smaller than threshold change vertex to this position
        cube.points[point_to_compute] = new_points[rho < rho_threshold].cpu()
        if verbose:
            print("cube.points", cube.points)

        if it % plot_each_it == 0 and plot_each_it > 0:
            p = pv.Plotter(window_size=[200, 100])
            p.add_mesh(cube, color="grey", show_edges=False)
            p.show()

        it += 1

    return cube
