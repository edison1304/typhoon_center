from utils.imports import *

def apply_rope_2d(x, base=10000):
    """
    x   : (B, C, H, W) 형태의 입력 텐서
    base: RoPE의 주파수(frequency) 계산 시 사용하는 base 상수 (기본 10000)
    
    반환값: (B, C, H, W) 형태의 텐서(2D RoPE가 적용된 결과)
    """
    B, C, H, W = x.shape
    assert C % 2 == 0, "채널 수 C는 2의 배수여야 합니다."
    half = C // 2  
    x1 = x[:, :half, :, :] 
    x2 = x[:, half:, :, :] 
    
    x1 = x1.reshape(B, half // 2, 2, H, W)  
    x1_even = x1[:, :, 0] 
    x1_odd  = x1[:, :, 1]  

    pos_h = torch.arange(H, device=x.device, dtype=x.dtype) 

    freq_seq = torch.arange(0, half, 2, device=x.device, dtype=x.dtype)
    denom = base ** (freq_seq / half) 
    
    angles_h = pos_h[:, None] / denom[None, :]
    sin_h = angles_h.sin()  
    cos_h = angles_h.cos()  

    sin_h = sin_h.transpose(0, 1).unsqueeze(0).unsqueeze(-1)  
    cos_h = cos_h.transpose(0, 1).unsqueeze(0).unsqueeze(-1)  
    sin_h = sin_h.expand(B, -1, -1, W) 
    cos_h = cos_h.expand(B, -1, -1, W) 

    x1_even_new = x1_even * cos_h - x1_odd * sin_h
    x1_odd_new  = x1_odd  * cos_h + x1_even * sin_h

    x1_new = torch.stack([x1_even_new, x1_odd_new], dim=2) 
    x1_new = x1_new.reshape(B, half, H, W)                

    x2 = x2.reshape(B, half // 2, 2, H, W)  
    x2_even = x2[:, :, 0] 
    x2_odd  = x2[:, :, 1] 

    pos_w = torch.arange(W, device=x.device, dtype=x.dtype)  
    
    angles_w = pos_w[:, None] / denom[None, :]
    sin_w = angles_w.sin()  
    cos_w = angles_w.cos() 

    sin_w = sin_w.transpose(0, 1).unsqueeze(0).unsqueeze(2) 
    cos_w = cos_w.transpose(0, 1).unsqueeze(0).unsqueeze(2) 
    sin_w = sin_w.expand(B, -1, H, -1) 
    cos_w = cos_w.expand(B, -1, H, -1) 

    x2_even_new = x2_even * cos_w - x2_odd * sin_w
    x2_odd_new  = x2_odd  * cos_w + x2_even * sin_w

    x2_new = torch.stack([x2_even_new, x2_odd_new], dim=2)  
    x2_new = x2_new.reshape(B, half, H, W)                 

    out = torch.cat([x1_new, x2_new], dim=1)  

    return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, expand_ratio=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm1 = LayerNormChannel(embed_dim)
        self.ffn = FeedForward(embed_dim, expand_ratio)
        self.norm2 = LayerNormChannel(embed_dim)

    def forward(self, x):
        x = apply_rope_2d(x)  
        attn_input = x
        b, c, h, w = x.shape
        flat_x = x.view(b, c, -1).permute(2, 0, 1)
        attn_output, _ = self.attn(flat_x, flat_x, flat_x)
        attn_output = attn_output.permute(1, 2, 0).view(b, c, h, w)
        x = attn_input + attn_output
        x = self.norm1(x)

        x2 = x
        b, c, h, w = x2.shape
        flat_x2 = x2.view(b, c, -1).permute(2, 0, 1)
        flat_x2 = self.ffn(flat_x2)
        x2 = flat_x2.permute(1, 2, 0).view(b, c, h, w)
        x = x + x2
        x = self.norm2(x)
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
    


class FeedForward(nn.Module):
    def __init__(self, dim, expand_ratio=2):
        super().__init__()
        hidden_dim = dim * expand_ratio
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)
    
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)))
    
class PatchEmbedding(nn.Module):
    def __init__(self, in_chans, embed_dim, patch_size, num_patches):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(
            in_chans,           
            embed_dim,         
            kernel_size=patch_size,
            stride=patch_size
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # 1. 이미지 -> 패치 임베딩 (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # 2. flatten & transpose (B, num_patches, embed_dim)
        return x

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
