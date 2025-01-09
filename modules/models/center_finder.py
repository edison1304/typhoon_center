from utils.imports import *
from ..components.CNNblock import main_ConvBlock, Downsample

class CenterFinder(nn.Module):
    def __init__(self, in_channels=4, hidden_dim=32):
        super().__init__()
        
        self.stem = nn.Sequential(
            main_ConvBlock(in_channels, hidden_dim, kernel_size=3, stride=1),
            main_ConvBlock(hidden_dim, hidden_dim, kernel_size=3, stride=1, use_residual=True)
        )
        
        self.stage1 = nn.Sequential(
            main_ConvBlock(hidden_dim, hidden_dim*2, kernel_size=3, stride=1, 
                          use_separable=True, use_residual=True),
        )
        
        self.down1 = Downsample(hidden_dim*2, hidden_dim*4)
        
        self.stage2 = nn.Sequential(
            main_ConvBlock(hidden_dim*4, hidden_dim*4, kernel_size=3, stride=1, 
                          use_separable=True, use_residual=True),
            main_ConvBlock(hidden_dim*4, hidden_dim*4, kernel_size=5, stride=1, 
                          use_separable=True, use_residual=True)
        )
        
        self.down2 = Downsample(hidden_dim*4, hidden_dim*8)
        
        self.stage3 = nn.Sequential(
            main_ConvBlock(hidden_dim*8, hidden_dim*8, kernel_size=3, stride=1, 
                          use_separable=True, use_residual=True),
            main_ConvBlock(hidden_dim*8, hidden_dim*8, kernel_size=5, stride=1, 
                          use_separable=True, use_residual=True),
            main_ConvBlock(hidden_dim*8, hidden_dim*8, kernel_size=5, stride=1, 
                          use_separable=True, use_residual=True),
            main_ConvBlock(hidden_dim*8, hidden_dim*8, kernel_size=13, stride=1, 
                          use_separable=True, use_residual=True),
            main_ConvBlock(hidden_dim*8, hidden_dim*8, kernel_size=25, stride=1, 
                          use_separable=True, use_residual=True)
        )
        
        self.down3 = Downsample(hidden_dim*8, hidden_dim*16)
        
        self.stage4 = nn.Sequential(
            main_ConvBlock(hidden_dim*16, hidden_dim*16, kernel_size=25, stride=1, 
                          use_separable=True, use_residual=True),
            main_ConvBlock(hidden_dim*16, hidden_dim*16, kernel_size=25, stride=1, 
                          use_separable=True, use_residual=True),
            main_ConvBlock(hidden_dim*16, hidden_dim*16, kernel_size=30, stride=1, 
                          use_separable=True, use_residual=False)
        )
        
        self.down4 = Downsample(hidden_dim*16, hidden_dim*32)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.center_head = nn.Sequential(
            nn.Linear(hidden_dim*32, hidden_dim*4),
            nn.GELU(),
            nn.Linear(hidden_dim*4, 2)  # Output (x, y) coordinates
        )
        
        self._initialize_weights()  # 가중치 초기화 호출

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.down3(x)
        x = self.stage4(x)
        x = self.down4(x)
        
        # Global pooling and coordinate prediction
        x = self.global_pool(x)
        x = x.flatten(1)
        coords = self.center_head(x)
        
        return coords
