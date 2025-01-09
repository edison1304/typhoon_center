from modules.components.Attentionblock import *


class TransformerRegressor(nn.Module):
    def __init__(self, in_chans=4, embed_dim=240, patch_size=10, num_heads=12, expand_ratio=4, depth=20):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, expand_ratio) for _ in range(depth)
        ])
        self.head = nn.Linear(embed_dim, 2)
        self.norm = LayerNormChannel(embed_dim)

    
    def forward(self, x):
        # 1) Conv2d를 통한 2D 임베딩 (B, embed_dim, 12, 12)
        x = self.patch_embed(x)
        # 2) TransformerBlock 반복 적용
        for blk in self.blocks:
            x = blk(x)
        x = x.mean(dim=[2,3]) 
        return self.head(x)
