import torch
from skimage.restoration import denoise_tv_chambolle
import numpy as np

class differenceCalculator:
    def __init__(self, min_difference, max_difference, tv_denoising: bool = False, tv_denoising_weight: float = 0.1):
        self.min_difference = min_difference
        self.max_difference = max_difference
        self.tv_denoising = tv_denoising
        self.tv_denoising_weight = tv_denoising_weight

    def __call__(self, image_t1, image_t2):
        difference  = image_t2 - image_t1
        
        if self.tv_denoising:
            difference_np = difference.squeeze().cpu().numpy()
            nan_mask = np.isnan(difference_np)
            difference_np[nan_mask] = 0

            denoised_difference_np = denoise_tv_chambolle(
                difference_np, weight=self.tv_denoising_weight
            )

            denoised_difference_np[nan_mask] = np.nan
            
            difference = torch.from_numpy(denoised_difference_np).to(device=difference.device, dtype=difference.dtype).unsqueeze(0)

        difference = torch.clip(difference, min=self.min_difference, max=self.max_difference)
        return difference
