import torch
import os


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
