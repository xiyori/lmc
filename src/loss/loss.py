# Taken from https://worksheets.codalab.org/worksheets/0x641008eb0b1b4768b865b58eddbe419c
# Sinkhorn Re-basin paper


import torch
import torch.nn as nn


# Eq 9 of the paper
class DistLoss(nn.Module):
    def forward(self, model0: nn.Module, model1: nn.Module):
        loss = 0
        num_params = 0
        for p0, p1 in zip(model0.parameters(), model1.parameters()):
            num_params += 1
            loss += (p0 - p1).abs().mean()
        loss /= num_params
        return loss


class MidLoss(nn.Module):
    def __init__(self, modela=None, criterion=None):
        super(MidLoss, self).__init__()

        self.criterion = criterion if criterion is not None else torch.nn.MSELoss()

        self.modela = modela
        for p in self.modela.parameters():
            p.requires_grad = False

    def set_model(self, modela):
        self.modela = modela
        for p in self.modela.parameters():
            p.requires_grad = False

    def forward(self, modelb, input, target):
        mid_lambda = torch.tensor([0.5]).to(input.device)

        for p1, p2 in zip(modelb.parameters(), self.modela.parameters()):
            p1.mul_(0.5)
            p1.add_(0.5 * p2.data)

        z = modelb(input)
        loss = self.criterion(z, target)

        return loss


class RndLoss(nn.Module):
    def __init__(self, modela=None, criterion=None):
        super(RndLoss, self).__init__()

        self.criterion = criterion if criterion is not None else torch.nn.MSELoss()

        self.modela = modela
        for p in self.modela.parameters():
            p.requires_grad = False

    def set_model(self, modela):
        self.modela = modela
        for p in self.modela.parameters():
            p.requires_grad = False

    def forward(self, modelb, input, target):
        random_l = torch.rand((1,)).to(input.device)

        for p1, p2 in zip(modelb.parameters(), self.modela.parameters()):
            p1.add_((random_l / (1 - random_l)) * p2.data)
            p1.mul_((1 - random_l))

        z = modelb(input)
        loss = self.criterion(z, target)

        return loss
