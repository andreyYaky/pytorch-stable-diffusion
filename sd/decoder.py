import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class VAE_AttentionBlock(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, Features, H, W)

        residue = x
        n, c, h, w = x.shape

        # (B, F, H, W) -> (B, F, H * W)
        x = x.view((n, c, h * w))

        # (B, F, H * W) -> (B, H * W, F)
        x = x.transpose(-1, -2)

        x = self.attention(x)

        # (B, H * W, F) -> (B, F, H * W)
        x = x.transpose(-1, -2)

        # (B, F, H * W) -> (B, F, H, W)
        x = x.view((n, c, h, w))

        x += residue

        return x

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, In_Channels, H, W)
        residue = x

        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residue)
    
# increase size of image and decrease number of channels

class VAE_Decoder(nn.Sequential):
    
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),

            nn.Conv2d(4, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 512),

            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            # (B, 512, H / 8, W / 8) -> (B, 512, H / 8, W / 8)
            VAE_ResidualBlock(512, 512),
            
            # (B, 512, H / 8, W / 8) -> (B, 512, H / 4, W / 4)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            # (B, 512, H / 4, W / 4) -> (B, 512, H / 2, W / 42)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),
            
            # (B, 256, H / 2, W / 2) -> (B, 256, H, W)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),

            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            # divide 128 features into groups of 32
            nn.GroupNorm(32, 128),
            
            nn.SiLU(),

            # (B, 128, H, W) -> (B, 3, H, W)
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4, H / 8, W / 8)
        
        # reverse constant scaling
        x /= 0.18215

        for module in self:
            x = module(x)

        # (B, 3, H, W)
        return x