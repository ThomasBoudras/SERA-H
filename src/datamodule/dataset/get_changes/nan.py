import torch

class getNan :
    def __call__(self, image_t1, image_t2) :
        return torch.tensor(torch.nan)