import torch

class treeCoverDetector:
    def __init__(self, tree_cover_threshold):
        self.tree_cover_threshold = tree_cover_threshold

    def __call__(self, image_t1, image_t2):
        binary_map_t1 = image_t1 > self.tree_cover_threshold
        binary_map_t2 = image_t2 > self.tree_cover_threshold
        return torch.logical_and(binary_map_t1, ~binary_map_t2)
