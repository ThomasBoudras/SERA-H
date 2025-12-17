class ratioThresholdDetector :
    def __init__(self, ratio_threshold, min_height_tree):
        self.ratio_threshold = ratio_threshold
        self.min_height_tree = min_height_tree

    def __call__(self, image_t1, image_t2):
        changes =  (image_t2 / (image_t1 + 1e-6)) < self.ratio_threshold
        changes[image_t1 < self.min_height_tree] = False #avoid the noise of tree less than 2 meters
        return changes