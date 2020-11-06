import torch
import numpy as np
import os
import warnings
import gc


def print_torch_mem_footprint():
    """Prints currently alive Tensors and Variables, from https://discuss.pytorch.org/t/how-to-debug-causes-of-gpu-memory-leaks/6741/2
    """
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass


def unpack_triangle_mesh(mesh_vertices, mesh_triangles):
    """Unpacks the encoded triangles from vertices and faces

    Args:
        mesh_vertices (np.array): Nx3 vertices
        mesh_triangles (np.array): Vx3 indices of respectively three vertices

    Returns:
        tuple of torch.tensor: (first_vertices,second_vertices,third_vertices)
    """
    mesh_vertices = torch.tensor(mesh_vertices).float()
    mesh_triangles = torch.tensor(mesh_triangles)

    # Unpack vertices
    v0 = torch.zeros([len(mesh_triangles), 3],
                     device=os.environ["TORCH_DEVICE"])
    v1 = torch.zeros([len(mesh_triangles), 3],
                     device=os.environ["TORCH_DEVICE"])
    v2 = torch.zeros([len(mesh_triangles), 3],
                     device=os.environ["TORCH_DEVICE"])
    for idx, t in enumerate(mesh_triangles):
        v0[idx] = mesh_vertices[t[0]]
        v1[idx] = mesh_vertices[t[1]]
        v2[idx] = mesh_vertices[t[2]]

    return (v0, v1, v2)


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


def fixRandomSeeds():
    """This function sets the random seeds in torch and numpy to enable reproducible behavior.
    """
    torch.manual_seed(42)
    np.random.seed(42)


def max_min_distance(points):
    """Computes the maximum of distance that each point in points has to its nearest neighbor."

    Args:
        points (torch tensor): Points to analyse.
    """
    distances = torch.zeros(len(points))
    for i in range(len(points)):
        dist = torch.norm(points - points[i], dim=1).float()
        dist[i] = 42424242  # distance to point itself
        distances[i] = torch.min(dist)
    return torch.max(distances).item()
