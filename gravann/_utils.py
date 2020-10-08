import torch

def max_min_distance(points):
    """Computes the maximum of distance that each point in points has to its nearest neighbor."

    Args:
        points (torch tensor): Points to analyse.
    """
    distances = torch.zeros(len(points))
    for i in range(len(points)):
        dist = torch.norm(points - points[i],dim=1).float()
        dist[i] = 42424242 #distance to point itself
        distances[i] = torch.min(dist)
    return torch.max(distances).item()