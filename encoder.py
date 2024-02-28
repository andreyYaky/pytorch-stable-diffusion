import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

# reduce size of image by increasing number of channels

class VAE_Encoder(nn.Sequential):

    def __init__(self):
        super().__init__(
            # (Batch_Size, Channel, H, W) -> (Batch_Size, 128, H, W)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),

            # (Batch_Size, 128, H, W) -> (Same)
            VAE_ResidualBlock(128, 128),

            # (Batch_Size, 128, H, W) -> (Same)
            VAE_ResidualBlock(128, 128),

            # (Batch_Size, 128, H, W) -> (Batch_Size, 128, H / 2, W / 2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

            # (Batch_Size, 128, H / 2, W / 2) -> (Batch_Size, 256, H / 2, W / 2)
            VAE_ResidualBlock(128, 256),

            # (Batch_Size, 256, H / 2, W / 2) -> (Same)
            VAE_ResidualBlock(256, 256),

            # (Batch_Size, 256, H / 2, W / 2) -> (Batch_Size, 256, H / 4, W / 4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),

            # (Batch_Size, 256, H / 4, W / 4) -> (Batch_Size, 512, H / 2, W / 2)
            VAE_ResidualBlock(256, 512),
            
            # (Batch_Size, 512, H / 4, W / 4) -> (Same)
            VAE_ResidualBlock(512, 512),
            
            # (Batch_Size, 512, H / 4, W / 4) -> (Batch_Size, 512, H / 8, W / 8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            
            # (Batch_Size, 512, H / 8, W / 8) -> (Same)
            VAE_ResidualBlock(512, 512),
            
            VAE_ResidualBlock(512, 512),
            
            VAE_ResidualBlock(512, 512),

            VAE_AttentionBlock(512),
            
            VAE_ResidualBlock(512, 512),

            nn.GroupNorm(32,512),
            
            nn.SiLU(),

            # (Batch_Size, 512, H / 8, W / 8) -> (Batch_Size, 8, H / 8, W / 8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            # (Batch_Size, 8, H / 8, W / 8) -> (Batch_Size, 8, H / 8, W / 8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )
    
    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, Channel=3, Height=512, Width=512)
        # noise: (Batch_Size, Outchannels, Height / 8, Width / 8)

        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                # (Padding_Left, Padding_Right, Padding_Top, Padding_Bottom)
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)
        
        # (Batch_Size, 8, Height, Height / 8, Width / 8) -> two tensors of shape (Batch_Size, 4, Height / 8, Width / 8)
        mean, log_variance = torch.chunk(x, 2, dim=1)

        # clamp extreme values
        log_variance = torch.clamp(log_variance, -30, 20)

        variance = log_variance.exp()

        stdev = variance.sqrt()

        # Z=N(0, 1) -> N(mean, variance)
        # X = mean + stdev * Z
        # sample from unit gaussian to new distribution with same Z value (basic stats)
        x = mean + stdev * noise

        # Scale the output by a constant
        x *= 0.18215

        return x
