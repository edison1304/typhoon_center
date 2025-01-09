from utils.imports import *
from timm.models.layers import DropPath

class main_ConvBlock(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size=3, 
        stride=1, 
        use_residual=False,
        use_separable=False,
        drop_path=0.2
    ):
        super(main_ConvBlock, self).__init__()
        padding = (kernel_size - 1) // 2
        
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,  # Changed from in_channels to out_channels
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False
        )
        self.norm = LayerNormChannel(out_channels)  # Changed to out_channels
        self.feedforward = FeedForward(out_channels, expand_ratio=4)  # Changed to out_channels
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_residual = use_residual
        self.project = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.project(x)
        h = self.depthwise(x)
        h = self.norm(h)
        h = self.feedforward(h)
        h = self.drop_path(h)
        if self.use_residual:
            h = h + identity
        return h
    
class FeedForward(nn.Module):
    def __init__(self, dim, expand_ratio):
        super().__init__()
        hidden_dim = dim * expand_ratio
        self.dim = dim
        self.W1 = nn.Conv2d(dim, hidden_dim, 1)
        self.W2 = nn.Conv2d(hidden_dim, dim, 1)
        self.W3 = nn.Conv2d(dim, hidden_dim, 1)

    def forward(self, x):
        return self.W2(self.W3(x) * torch.sigmoid(self.W1(x)))

class Downsample(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=2, stride=2):
        super().__init__()
        padding = (kernel_size - 1) // 2 if stride == 1 else 0
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding=padding)
        self.norm = LayerNormChannel(out_dim)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x

class LayerNormChannel(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps = 1e-5

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / (var + self.eps).sqrt()
        return x * self.weight + self.bias

