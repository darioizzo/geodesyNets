import torch
import numpy as np
import os
import warnings
import gc
import pickle as pk


def print_torch_mem_footprint():
    """Prints currently alive Tensors and Variables, from https://discuss.pytorch.org/t/how-to-debug-causes-of-gpu-memory-leaks/6741/2
    """
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass


def get_asteroid_bounding_box(asteroid_pk_path):
    """Computes a rectangular cuboid bounding box for the mesh of the sample.

    Args:
        asteroid_pk_path (str): Path to mesh of target asteroid

    Returns:
        np.array: domain as [min,max]^3
    """
    with open(asteroid_pk_path, "rb") as file:
        mesh_vertices, _ = pk.load(file)
    mesh_vertices = np.array(mesh_vertices)
    box = [[np.min(mesh_vertices[:, 0]), np.max(mesh_vertices[:, 0])],
           [np.min(mesh_vertices[:, 1]), np.max(mesh_vertices[:, 1])],
           [np.min(mesh_vertices[:, 2]), np.max(mesh_vertices[:, 2])]]

    return box


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

def is_quadratic_param(param_name: str) -> bool:
    return (
        param_name.endswith(".weight_g") or
        param_name.endswith(".weight_b") or
        param_name.endswith(".bias_g") or
        param_name.endswith(".c")
    )


class EarlyStopping():
    """A rudimentary implementation of callback that tells you when to early stop
    """

    def __init__(self, save_folder, patience=2000, warmup=3000):
        """Rudimentary EarlyStopping implementation

        Args:
            savefolder (str): Path where best model should be stored
            patience (int, optional): After how many calls without improvement the stopper will return True (implies to stop). Defaults to 100.
            warmup (int, optional): Early stopping will not trigger in the warmup.
        """
        self.minimal_loss = 424242424242
        self.save_folder = save_folder
        self.warmup = warmup
        self.patience = patience
        self._it = 0
        self._iterations_without_improvement = 0

    def early_stop(self, loss_value, model):
        """Update the early stopper. Returns true if patience was reach without improvement

        Args:
            loss_value (float): current loss value
            model (torch.nn): current model, always save best checkpoint

        Returns:
            bool: True if patience reach, else False
        """

        self._it += 1

        if loss_value < self.minimal_loss:
            self.minimal_loss = loss_value
            torch.save(model.state_dict(), self.save_folder + "best_model.mdl")
            self._iterations_without_improvement = 0
        else:
            self._iterations_without_improvement += 1

        if self._it < self.warmup:
            return False

        if self._iterations_without_improvement >= self.patience:
            return True
        else:
            return False
