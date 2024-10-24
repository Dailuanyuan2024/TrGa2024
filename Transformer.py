import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

MIN_NUM_PATCHES = 16

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs): #x=b*n+1*c
        return self.fn(self.norm(x), **kwargs) #层归一化

class FeedForward(nn.Module): #进行了一次全连接 --> GELU --> 全连接的变换
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(), #高斯误差线性单元函数 GELUs其实是dropout、zoneout、ReLU的综合，GELU对于输入乘以一个 [ 0 , 1 ] [0,1] [0,1]组成的mask，而该mask的生成则是依概率随机的依赖于输入。假设输入为 X X X, mask为 m m m，则 m m m服从一个伯努利分布 Φ ( x ) = P ( X < x ) \Phi(x)=P(X<x) Φ(x)=P(X<x)，其中 X X X服从标准正态分布。这么选择是因为神经元的输入趋向于正态分布，这么设定使得当输入 x x x减小的时候，输入会有一个更高的概率被dropout掉，这样的激活变换就会随机依赖于输入了。
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64): #这里的heads=8是不是应该改一下呀
        super().__init__()
        inner_dim = dim_head * heads #dim_head=64 heads=2
        self.heads = heads
        self.scale = dim ** -0.5 #dim =c=128

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False) #Linear(in_features=128, out_features=384, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
        )

    def forward(self, x, mask = None): 
        b, n, _, h = *x.shape, self.heads #x=b*n+1*c 
        qkv = self.to_qkv(x).chunk(3, dim = -1) #3个b*n+1*c 
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv) #3个bh*n+1*c/2

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale #bh*n+1*n+1
        mask_value = -torch.finfo(dots.dtype).max #-3.4028234663852886e+38

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1) #[2, 2, 251, 251]

        out = torch.einsum('bhij,bhjd->bhid', attn, v) #[2, 2, 2001, 64]
        out = rearrange(out, 'b h n d -> b n (h d)') #[2, 2001, 128]
        out = self.to_out(out) #[2, 2001, 128]
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth): #depth为Transformer Encoder的堆叠次数，也即该部分深度，我们使用ModuleList既保持代码整洁又实现了模块堆叠
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim)))
            ]))
    def forward(self, x, mask = None):

        for attn, ff in self.layers:
            x = attn(x, mask = mask) #[2, 251, 128]
            x = ff(x) #[2, 2001, 128]
        return x
