import torch

class relativeDifferenceCalculator:
    def __init__(self, max_relative_difference):
        self.max_relative_difference = max_relative_difference

    def __call__(self, image_t1, image_t2):
        difference  = image_t2 - image_t1
        relative_difference = difference / (image_t1 + 1)
        relative_difference = torch.clip(relative_difference, min=-1, max=self.max_relative_difference) # Since in the worst case images_t2 is zero, the min is -1
        return relative_difference