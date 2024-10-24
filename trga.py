import torch
import torch.nn as nn
from loss import batch_episym
from Transformer import Transformer
from einops import rearrange, repeat

    
class ResNet_Block(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(ResNet_Block, self).__init__()
        self.right = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
        )
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
        )

    def forward(self, x):
        x1 = self.right(x) 
        out = self.left(x)
        out = out + x1
        return torch.relu(out)
    
class CGNS_Block(nn.Module):
    def __init__(self, in_channel):
        super(CGNS_Block, self).__init__()
        self.in_channel = in_channel
        self.conv = nn.Sequential(
            nn.Conv2d(2*self.in_channel, self.in_channel, (1, 1)),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True),
        )
        # self.conv1 = nn.Conv2d(2*self.in_channel, self.in_channel, (1, 1))

    def attention(self, w):
        w = torch.relu(torch.tanh(w)).unsqueeze(-1) #w[32,2000,1] 变成0到1的权重
        A = torch.bmm(w, w.transpose(1, 2)) #A[32,1,1]
        return A

    def graph_aggregation(self, x, w):
        B, _, N, _ = x.size() #B=32,N=2000
        with torch.no_grad():
            A = self.attention(w) #A[32,1,1]
            I = torch.eye(N).unsqueeze(0).to(x.device).detach() #I[1,2000,2000]单位矩阵
            A = A + I #A[32,2000,2000]
            D_out = torch.sum(A, dim=-1) #D_out[32,2000]
            D = (1 / D_out) ** 0.5
            D = torch.diag_embed(D) #D[32,2000,2000]
            L = torch.bmm(D, A)
            L = torch.bmm(L, D) #L[32,2000,2000]
        out1 = x.squeeze(-1).transpose(1, 2).contiguous() #out[32,2000,128]
        out = x.squeeze(-1)
        out0 = torch.bmm(out, A).unsqueeze(-1)
        out1 = torch.bmm(L, out1).unsqueeze(-1)
        out1 = out1.transpose(1, 2).contiguous() #out[32,128,2000,1]
        out = torch.cat((out0,out1),dim=1)
        

        return out

    def forward(self, x, w):
        out = self.graph_aggregation(x, w)
        out = self.conv(out)
        return out

class TransM(nn.Module):
    def __init__(self, in_channel, out_channel, p_size,  T_depth, heads, dim_head, mlp_dim):
        super(TransM, self).__init__()
        self.p_size = p_size #1
        self.patch_to_embedding = nn.Linear(in_channel, out_channel) 
        self.cls_token = nn.Parameter(torch.randn(1, 1, out_channel)) 
        self.transformer = Transformer(out_channel, T_depth, heads, dim_head, mlp_dim)

    def forward(self, x):
        _,_,hh,ww = x.size() 
        """
        这两步和一起是Vision Transformer的patch_embedding
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.p_size, p2=self.p_size) #bnc
        x = self.patch_to_embedding(x) #bnc，对维度进行缩放，用nn.Linear就可以
        """
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.p_size, p2=self.p_size) 
        x = self.patch_to_embedding(x) 
        b, n, _ = x.size()
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b) 
        x = torch.cat((cls_tokens, x), dim=1) 
        x = self.transformer(x)  
        x = rearrange(x[:, 1:], 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=self.p_size, p2=self.p_size, h=hh, w=ww) 
        return x 

class Trans_gd(nn.Module):
    def __init__(self, net_channels):
        nn.Module.__init__(self)
        channels = net_channels
        super(Trans_gd, self).__init__()
        self.trans1 = TransM(in_channel=128, out_channel=128, p_size=1,  T_depth=2,
                             heads=4, dim_head=32, mlp_dim=128)
        self.gsda = CGNS_Block(channels)


        self.embed_01 = nn.Sequential(
            ResNet_Block(channels, channels),
        )
        self.linear_01 = nn.Conv2d(channels, 1, (1, 1))
    

    def forward(self, x):
        batch_size = x.shape[0]           
        x0 = self.trans1(x)     
        out0 = self.embed_01(x0)
        w0 = self.linear_01(out0).view(batch_size, -1) 
        out_g = self.gsda(out0, w0.detach()) 
        out = out_g + out0


        return out

class TRGA_Block(nn.Module):
    def __init__(self, net_channels, input_channel):
        nn.Module.__init__(self)
        channels = net_channels
        super(TRGA_Block, self).__init__()
        # self.in_channel = 4 if self.initial is True else 6


        self.conv = nn.Sequential(
            nn.Conv2d(input_channel, channels, (1, 1)), #4或6 → 128
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        
        self.l0 = []
        for _ in range(4):
            self.l0.append(TransM(in_channel=128, out_channel=128, p_size=1,  T_depth=2,
                             heads=4, dim_head=32, mlp_dim=128))
        
        self.l0 = nn.Sequential(*self.l0)
        self.l1 = Trans_gd(channels)

        self.embed_1 = nn.Sequential(
            ResNet_Block(channels, 2*channels),
        )
        self.linear_1 = nn.Conv2d(2*channels, 1, (1, 1))
    

    def forward(self, data, xs):
        batch_size, num_pts = data.shape[0], data.shape[2]
        
        out = self.conv(data) 
 
        x0 = self.l0(out) 
        x1 = self.l1(x0)
        out1 = self.embed_1(x1)


        logits = torch.squeeze(torch.squeeze(self.linear_1(out1),3),1)
        e_hat = weighted_8points(xs, logits)

        x1, x2 = xs[:,0,:,:2], xs[:,0,:,2:4]
        e_hat_norm = e_hat
        residual = batch_episym(x1, x2, e_hat_norm).reshape(batch_size, 1, num_pts, 1)

        return logits, e_hat, residual


class TRGA(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.iter_num = config.iter_num
        # depth_each_stage = config.net_depth//(config.iter_num+1)
        self.side_channel = (config.use_ratio==2) + (config.use_mutual==2)
        self.weights_init = TRGA_Block(config.net_channels, 4+self.side_channel)
        self.weights_iter = [TRGA_Block(config.net_channels, 6+self.side_channel) for _ in range(config.iter_num)]
        self.weights_iter = nn.Sequential(*self.weights_iter)
        

    def forward(self, data):
        assert data['xs'].dim() == 4 and data['xs'].shape[1] == 1
        batch_size, num_pts = data['xs'].shape[0], data['xs'].shape[2]
        input = data['xs'].transpose(1,3)
        if self.side_channel > 0:
            sides = data['sides'].transpose(1,2).unsqueeze(3)
            input = torch.cat([input, sides], dim=1)

        res_logits, res_e_hat = [], []
        logits, e_hat, residual = self.weights_init(input, data['xs']) 
        res_logits.append(logits), res_e_hat.append(e_hat)
        for i in range(self.iter_num):
            logits, e_hat, residual = self.weights_iter[i](
                torch.cat([input, residual.detach(), torch.relu(torch.tanh(logits)).reshape(residual.shape).detach()], dim=1),
                data['xs'])
            res_logits.append(logits), res_e_hat.append(e_hat)
        return res_logits, res_e_hat  

#############################################################
def batch_symeig(X):
    # it is much faster to run symeig on CPU
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b,d,d)
    for batch_idx in range(X.shape[0]):
        e,v = torch.symeig(X[batch_idx,:,:].squeeze(), True)
        bv[batch_idx,:,:] = v
    bv = bv.cuda()
    return bv


def weighted_8points(x_in, logits):
    # x_in: batch * 1 * N * 4
    x_shp = x_in.shape
    # Turn into weights for each sample
    weights = torch.relu(torch.tanh(logits))
    x_in = x_in.squeeze(1)
    
    # Make input data (num_img_pair x num_corr x 4)
    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1)

    # Create the matrix to be used for the eight-point algorithm
    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1)
    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1), wX)
    

    # Recover essential matrix from self-adjoing eigen
    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))

    # Make unit norm just in case
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat

        