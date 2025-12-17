import torch

class getNan :
    def __call__(self, bounds, row, transform) :
        return torch.tensor(torch.nan), {}