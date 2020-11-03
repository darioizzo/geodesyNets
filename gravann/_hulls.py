from scipy.spatial import Delaunay
import numpy as np
import torch
import os
from copy import deepcopy


def alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.

    Args:
        points (np.array of shape (n,2)): points.
        alpha (float): alpha value.
        only_outer (bool): specifies if we keep only the outer border
        or also inner edges.

    Returns:
        set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
        the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"
    assert points.shape[1] == 2, "Need two dimensional points"

    def add_edge(edges, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges


def ray_triangle_intersect(ray_o, ray_d, v0, v1, v2):
    """Möller–Trumbore intersection algorithm

    Computes whether a ray intersect a triangle

    Args:
        ray_o (3D np.array): origin of the ray.
        ray_d (3D np.array): direction of the ray.
        v0, v1, v2 (3D np.array): triangle vertices

    Returns:
        boolean value if the intersection exist (includes the edges)

    See: https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
    """
    if ray_o.shape != (3,):
        raise ValueError("Shape f ray_o input should be (3,)")
    edge1 = v1-v0
    edge2 = v2-v0
    h = np.cross(ray_d, edge2)

    a = np.dot(edge1, h)

    if a < 0.000001 and a > -0.000001:
        return False

    f = 1.0 / a
    s = ray_o-v0
    u = np.dot(s, h) * f

    if u < 0 or u > 1:
        return False

    q = np.cross(s, edge1)
    v = np.dot(ray_d, q) * f

    if v < 0 or u + v > 1:
        return False

    t = f * np.dot(edge2, q)

    return t > 0


def rays_triangle_intersect(ray_o, ray_d, v0, v1, v2):
    """Möller–Trumbore intersection algorithm (vectorized)

    Computes whether a ray intersect a triangle

    Args:
        ray_o ((N, 3) np.array): origins for the ray.
        ray_d (3D np.array): direction of the ray.
        v0, v1, v2 (3D np.array): triangle vertices

    Returns:
        boolean value if the intersection exist (includes the edges)

    See: https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
    """
    if ray_o.shape[1] != 3:
        raise ValueError(
            "Shape f ray_o input should be (N, 3) in this vectorized version")
    edge1 = v1-v0
    edge2 = v2-v0
    h = np.cross(ray_d, edge2)

    a = np.dot(edge1, h)

    if a < 0.000001 and a > -0.000001:
        return [False]*len(ray_o)

    f = 1.0 / a
    s = ray_o-v0
    u = np.dot(s, h) * f

    crit1 = np.logical_not(np.logical_or(u < 0, u > 1))
    q = np.cross(s, edge1)
    v = np.dot(q, ray_d) * f
    crit2 = np.logical_not(np.logical_or(v < 0, u+v > 1))
    t = f * np.dot(q, edge2)
    crit3 = t > 0

    return np.logical_and(np.logical_and(crit1, crit2), crit3)


def rays_triangle_intersect_torch(ray_o, ray_d, v0, v1, v2):
    """Möller–Trumbore intersection algorithm (vectorized)

    Computes whether a ray intersect a triangle

    Args:
        ray_o ((3) torch.tensor): origins for the ray.
        ray_d (3D torch.tensor): direction of the ray.
        v0, v1, v2 (Nx3 torch.tensor): triangle vertices

    Returns:
        boolean value if the intersection exist (includes the edges)

    See: https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
    """
    V = len(v0)

    edge1 = v1-v0
    edge2 = v2-v0
    ray_o = ray_o.repeat(V, 1)
    h = torch.cross(ray_d.expand(V, 3), edge2)

    a = torch.einsum('bs,bs->b', edge1, h)

    f = 1.0 / a
    s = ray_o - v0
    u = torch.einsum('bs,bs->b', s, h) * f

    crit1 = torch.logical_not(torch.logical_or(u < 0.0, u > 1.0))
    q = torch.cross(s, edge1)

    v = torch.einsum('bs,bs->b', q,
                     ray_d.expand(V, 3)) * f
    crit2 = torch.logical_not(torch.logical_or(v < 0.0, u+v > 1.0))
    t = f * torch.einsum('bs,bs->b', q,
                         edge2)

    crit3 = t > 0.0
    result = torch.logical_and(torch.logical_and(crit1, crit2), crit3)

    # Set those 0 where a < 0.000001 or a > -0.000001
    result[torch.logical_and(a < 0.0000001, a > -0.0000001)] = 0
    return torch.sum(result)


def is_outside_torch(points, triangles):
    """Detects if points are outside a 3D mesh

    Args:
        points ((N,3)) torch.tensor): points to test.
        mesh_vertices ((3,M,3) torch.tensor): vertices pf the mesh
        mesh_triangles ((M,3) torch.tensor): ids of each triangle

    Returns:
        torch.tensor of boolean values determining whether the points are inside
    """
    counter = torch.zeros([len(points)], device=os.environ["TORCH_DEVICE"])
    direction = torch.tensor([0., 0., 1.], device=os.environ["TORCH_DEVICE"])
    v0, v1, v2 = triangles
    for idx, point in enumerate(points):
        counter[idx] = rays_triangle_intersect_torch(
            point, direction, v0, v1, v2)

    return (counter % 2) == 0


def is_outside(points, mesh_vertices, mesh_triangles):
    """Detects if points are outside a 3D mesh

    Args:
        points ((N,3)) np.array): points to test.
        mesh_vertices ((M,3) np.array): vertices pf the mesh
        mesh_triangles ((M,3) np.array): ids of each triangle

    Returns:
        np.array of boolean values determining whether the points are inside
    """
    counter = np.array([0]*len(points))
    direction = np.array([0, 0, 1])
    for t in mesh_triangles:
        counter += rays_triangle_intersect(
            points, direction, mesh_vertices[t[0]], mesh_vertices[t[1]], mesh_vertices[t[2]])
    return (counter % 2) == 0


def is_inside(points, mesh_vertices, mesh_triangles):
    """Detects if points are inside a 3D mesh

    Args:
        points ((N,3)) np.array): points to test.
        mesh_vertices ((M,3) np.array): vertices pf the mesh
        mesh_triangles ((M,3) np.array): ids of each triangle

    Returns:
        np.array of boolean values determining whether the points are inside
    """
    counter = np.array([0]*len(points))
    direction = np.array([0, 0, 1])
    for t in mesh_triangles:
        counter += rays_triangle_intersect(
            points, direction, mesh_vertices[t[0]], mesh_vertices[t[1]], mesh_vertices[t[2]])
    return (counter % 2) == 1
