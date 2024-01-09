"""
This file contains the generator part of the GAN model.
"""

import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, inp_dim: tuple[int], n_channels: int=1, n_conv_kernels: int=3, device='cpu'):
        """
        The constructor of the Generator part of the network.

        inp_dim: the dimensions of the input
        n_channels: the number of channels in the input. defaults to 1
        n_conv_kernels: the number of convolutional kernels in the first layer of the network. defaults to 3  
        device: the device to send the calculations to. defaults to 'cpu' 
        """
        super(Generator, self).__init__()  

        self.G = nn.Sequential(
            nn.ConvTranspose2d(n_channels, n_conv_kernels, 2, device=device),

            nn.LeakyReLU(0.1, True),

            nn.Dropout(inplace=True),

            nn.Linear(n_conv_kernels, n_conv_kernels, device=device),
            nn.LeakyReLU(0.1, True),

            nn.ConvTranspose2d(n_conv_kernels, n_channels),
            nn.Tanh()
        )

    def forward(self, X: torch.Tensor):
        """
        The forward pass of the network.

        returns an image.
        """
        return self.G(X) 
