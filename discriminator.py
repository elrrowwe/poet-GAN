"""
This file contains the discriminator part of the network.
"""

import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, inp_dim: tuple[int], n_channels=1):
        """
        The constructor of the Discriminator part of the network.
        
        inp_dim: the dimensions of the input
        n_channels: the number of channels in the input. defaults to 1
        """

        self.D = nn.Sequential(
            
        )