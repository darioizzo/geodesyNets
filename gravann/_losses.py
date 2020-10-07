import torch

def normalized_loss(predicted, labels):
    c = sum(torch.mul(labels, predicted))/sum(torch.pow(predicted,2))
    return sum(torch.pow(torch.sub(labels,c*predicted),2)) / len(labels)


def mse_loss(predicted, labels):
    return torch.nn.MSELoss(predicted, labels)