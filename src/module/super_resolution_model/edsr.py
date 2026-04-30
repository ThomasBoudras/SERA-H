"""
EDSR (Enhanced Deep Residual Networks for Single Image Super-Resolution) implementation.

This architecture is adapted from the official PyTorch implementation of the original paper:
https://github.com/sanghyun-son/EDSR-PyTorch

Lim, Bee, et al. "Enhanced deep residual networks for single image super-resolution."
Proceedings of the IEEE conference on computer vision and pattern recognition workshops. 2017.
"""
import torch.nn as nn
import torch
import numpy as np
from src.module.super_resolution_model.components.edsr_utils import ResBlock, Upsampler, conv2d
from pathlib import Path

from src import global_utils as utils

log = utils.get_logger(__name__)

class EDSR(nn.Module):
    def __init__(
            self,
            n_resblocks,
            scale,
            n_feats,
            n_channels,
            res_scale,
            pretrained_model_path,
            input_type,
            freeze_model,
        ):
        super(EDSR, self).__init__()
        self.forward_method = self.forward_timeseries if input_type == "TIMESERIES" else self.forward_composites
             
        kernel_size = 3 
        act = nn.ReLU(True)

        self.n_channels = n_channels
        self.scale = scale
        self.pretrained_model_path = Path(pretrained_model_path).resolve() if pretrained_model_path is not None else None 
        self.input_type = input_type

        # define head module
        m_head = [conv2d(n_channels, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv2d, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv2d(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            Upsampler(conv2d, scale, n_feats, act=False),
            conv2d(n_feats, n_channels, kernel_size)
        ]
        
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        
        if self.pretrained_model_path is not None :
            self.load_partial_weight()

        if freeze_model:
            self.freeze_model()

    def freeze_model(self):
        for param in self.head.parameters():
            param.requires_grad = False
        for param in self.body.parameters():
            param.requires_grad = False
        for param in self.tail.parameters():
            param.requires_grad = False

    def load_partial_weight(self) :
        log.info(f"Using the pre-trained model {self.pretrained_model_path.name} to initialise the model")
        load_from = torch.load(self.pretrained_model_path,  map_location=torch.device('cpu'), weights_only=True)

        # We only change weight of the head and tail of the model, the body does not need to be changed

        # Update "head.0.weight"
        module_tensor = load_from["head.0.weight"]
        mean_weight = module_tensor.mean(dim=1, keepdim=True)
        expanded_weight = mean_weight.expand(-1, self.n_channels, -1, -1).clone()
        expanded_weight[:, :3, :, :] = module_tensor[:, [2,1,0], :, :]  # Sentinel-2 start with BGR
        load_from["head.0.weight"] = expanded_weight

        # Update "tail.1.weight"
        module_tensor = load_from["tail.1.weight"]
        mean_weight = module_tensor.mean(dim=0, keepdim=True)
        expanded_weight = mean_weight.expand(self.n_channels, -1, -1, -1).clone()
        expanded_weight[:3, :, :, :] = module_tensor[[2,1,0], :, :, :]  # Sentinel-2 start with BGR
        load_from["tail.1.weight"] = expanded_weight

        # Update "tail.1.bias"
        module_tensor = load_from["tail.1.bias"]
        mean_bias = module_tensor.mean().unsqueeze(0)
        expanded_bias = mean_bias.expand(self.n_channels).clone()
        expanded_bias[:3] = module_tensor[[2,1,0]] # Sentinel-2 start with BGR
        load_from["tail.1.bias"] = expanded_bias

        load_from["tail.0.0.weight"] = load_from["tail.0.0.weight"][:256*self.scale, :, :, :]
        load_from["tail.0.0.bias"] = load_from["tail.0.0.bias"][:256*self.scale]
        load_from["tail.0.2.weight"] = load_from["tail.0.2.weight"][:256*self.scale, :, :, :]
        load_from["tail.0.2.bias"] = load_from["tail.0.2.bias"][:256*self.scale]

        self.load_state_dict(load_from, strict=False)
           
    def forward_timeseries(self, inputs, targets, metadata):
        # inputs: (B, T, C, H, W)
        B, T, C, H, W = inputs.shape
        inputs = inputs.view(B * T, C, H, W)
        inputs = self.head(inputs)
        res = self.body(inputs)
        res += inputs
        inputs = self.tail(res)
        inputs = inputs.view(B, T, self.n_channels, inputs.shape[-2], inputs.shape[-1])
        return inputs

    def forward_composites(self, inputs, targets, metadata):
        # inputs: (B, C, H, W)
        inputs = self.head(inputs)
        res = self.body(inputs)
        res += inputs
        inputs = self.tail(res)
        return inputs 
    
    def forward(self, inputs, targets, metadata) :
        return self.forward_method(inputs, targets, metadata)


        


