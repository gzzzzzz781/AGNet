import time
import torch
import torch.nn as nn
from thop import profile
import torch.nn.functional as F
from einops import rearrange

def gradient(img):
    diff_w = torch.abs(torch.diff(img, dim=3))
    diff_w = F.pad(diff_w, (0, 1, 0, 0))

    diff_h = torch.abs(torch.diff(img, dim=2))
    diff_h = F.pad(diff_h, (0, 0, 0, 1))

    tv = diff_w + diff_h

    return tv

class LConv(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1, dilation=1, kernel_size=3, bias=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, in_dim, kernel_size=kernel_size, padding=dilation, stride=stride, groups=in_dim, bias=bias, dilation=dilation)
        self.conv2 = nn.Conv2d(in_dim, out_dim, kernel_size=1, padding=0, bias=bias)

    def forward(self, x):
        return self.conv2(self.conv1(x))

#GFFN
class Gated_Feed_Forward_Network(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj1 = LConv(in_dim=dim, out_dim=dim)
        self.proj2 = LConv(in_dim=dim, out_dim=dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        x1 = self.proj1(x)
        x2 = self.proj2(x)
        x2 = x2 * self.gelu(x1)

        return x2

#GSAB
class Global_Spitial_Attention_Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_q = LConv(in_dim=dim, out_dim=2)
        self.conv_k = LConv(in_dim=dim, out_dim=2)
        self.conv_v = LConv(in_dim=dim, out_dim=dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        b, c, h, w = x.shape

        q, k = rearrange(self.conv_q(x), 'b (head c) h w -> b head c (h w)', head=1), \
               rearrange(self.conv_k(x), 'b (head c) h w -> b head c (h w)', head=1)
        v = rearrange(self.conv_v(x), 'b (head c) h w -> b head c (h w)', head=1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        value = torch.matmul(v, q.transpose(-2, -1))
        out = torch.matmul(value, k)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=1, h=h, w=w)

        return out

#GCAB
class Gradient_guided_Channel_Attention_Block(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.conv_q = LConv(in_dim=2*dim, out_dim=dim)
        self.conv_k = LConv(in_dim=2*dim, out_dim=dim)
        self.conv_v = LConv(in_dim=dim, out_dim=dim)

    def forward(self, x):
        b, c, h, w = x.shape

        x1 = gradient(x)
        x2 = torch.cat([x, x1], dim=1)
        q, k = rearrange(self.conv_q(x2), 'b (head c) h w -> b head c (h w)', head=self.num_heads), \
               rearrange(self.conv_k(x2), 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(self.conv_v(x), 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        return out

class LightTransformer(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.atten = Global_Spitial_Attention_Block(dim=dim)
        self.mlp1 = Gated_Feed_Forward_Network(dim=dim)
        self.atten2 = Gradient_guided_Channel_Attention_Block(dim=dim, num_heads=num_heads)
        self.mlp2 = Gated_Feed_Forward_Network(dim=dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        x = x.permute(0, 2, 3, 1)  # BCHW → BHWC
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # BHWC → BCHW

        x1 = self.atten(x)
        x1 = x1 + x

        x2 = self.mlp1(x1)
        x2 = x2 + x1

        x3 = self.atten2(x2)
        x3 = x3 + x2

        x4 = self.mlp2(x3)
        x4 = x4 + x3

        return x4


class AGNet(nn.Module):
    def __init__(self, in_chans=3, embed_dim=64, num_heads=8):
        super().__init__()

        self.conv1 = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.layers = nn.ModuleList([LightTransformer(dim=embed_dim, num_heads=num_heads) for _ in range(18)])
        self.conv_last = nn.Conv2d(embed_dim, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.convs = nn.ModuleList([LConv(in_dim=embed_dim, out_dim=embed_dim) for _ in range(3)])

        self.LConv1 = LConv(in_dim=embed_dim, out_dim=embed_dim)
        self.LConv2 = LConv(in_dim=embed_dim, out_dim=embed_dim)
        self.LConv3 = LConv(in_dim=embed_dim, out_dim=embed_dim)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU(negative_slope=0.02)

    def forward(self, x):
        x1 = self.conv1(x)

        y1 = self.convs[0](x1)
        y1 = self.relu(y1)

        for i in range(6):
            x1 = self.layers[i](x1)
        x2 = self.LConv1(x1)
        y1 = y1 + x2

        y2 = self.convs[1](y1)
        y2 = self.relu(y2)

        for i in range(6, 12):
            x2 = self.layers[i](x2)
        x3 = self.LConv2(x2)
        y2 = y2 + x3

        y3 = self.convs[2](y2)
        y3 = self.relu(y3)

        for i in range(12, 18):
            x3 = self.layers[i](x3)
        x4 = self.LConv3(x3)

        x4 = x4 + y3

        x5 = self.conv_last(x4)
        x5 = self.sigmoid(x5)

        return x5

def flops():
    height, width = 128, 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x1 = torch.randn((1, 3, height, width)).to(device)

    model = AGNet().to(device)
    model.eval()

    start_time = time.time()
    for i in range(200):
        with torch.no_grad():
            x = model(x1)
    end_time = time.time()
    print(x.max().item(), x.min().item())


    print("Infer time: {}".format(end_time-start_time))

    flops, params = profile(model, inputs=(x1,), verbose=False)
    print(f'model(HDR-Transformer): flops: {flops / 1e9:.3f} G, params: {params / 1e6:.3f} M')

if __name__ == '__main__':
    flops()
