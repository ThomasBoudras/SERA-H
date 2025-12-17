class differenceThresholdDetector :
    def __init__(self, difference_threshold):
        self.difference_threshold = difference_threshold

    def __call__(self, image_t1, image_t2):
        difference = image_t2 - image_t1
        return difference < self.difference_threshold