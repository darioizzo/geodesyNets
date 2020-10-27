import torch


def normalized_loss(predicted, labels):
    c = torch.sum(torch.mul(labels, predicted)) / \
        torch.sum(torch.pow(predicted, 2))
    return torch.sum(torch.pow(torch.sub(labels, c*predicted), 2)) / len(labels)


def mse_loss(predicted, labels):
    return torch.nn.MSELoss()(predicted, labels)
