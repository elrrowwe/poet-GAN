"""
This file contains the generator part of the GAN model.

The generator, in this context, accepts a 1 x z_dim (see the constructor) vector of noise drawn from the normal distribution
and gradually upsamples it to the desired size, while also transforming it into a realistic image. 
"""

import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim: int, lin1_hw: int=5, lin1_channels: int=256, output_channels: int=1, device: str='cpu'):
        """
        The constructor of the Generator part of the network.

        z_dim: the dimension of the input noise vector
        lin1_hw: the height, width of the feature map produced by the first, dense layer. defaults to 5
        lin1_channels: the number of channels in the output of the first, dense layer. defaults to 128
        output_channels: the number of channels in the output of the network. defaults to 1 
        device: the device to send the calculations to. defaults to 'cpu' 
        """
        super(Generator, self).__init__()  

        self.z_dim = z_dim
        self.lin1_channels = lin1_channels
        self.lin1_hw = lin1_hw
        self.output_channels = output_channels
        self.device = device

        #converting the noise vector into a tensor
        self.lin1 = nn.Linear(self.z_dim, self.lin1_hw * self.lin1_hw * self.lin1_channels, device=device), #a lin1_hw x lin1_hw spatial resolution, lin1_channels channels. the resolution may be in the lower range for efficiency 

        self.G = nn.Sequential(
            #the upsampling sequence 
            nn.ConvTranspose2d(self.lin1_channels, self.lin1_channels / 2, kernel_size=4, stride=2, device=self.device),
            nn.LeakyReLU(0.1, True),

            nn.ConvTranspose2d(self.lin1_channels / 2, self.lin1_channels / 4, kernel_size=4, stride=2, device=self.device),
            nn.LeakyReLU(0.1, True),

            nn.ConvTranspose2d(self.lin1_channels / 4, output_channels, kernel_size=4, stride=2, device=self.device),

            nn.Tanh()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the network.

        returns an image.
        """
        z = self.lin1(z)

        z = z.view(-1, 256, 4, 4)
        
        return self.G(z) 
