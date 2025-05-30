import torch
import torch.nn as nn
import torch.nn.functional as F

class SSA(nn.Module):
    def __init__(self, in_dim, num_heads, bias=False, 
                 attn_drop_prob=0., proj_drop_prob=0.):
        super(SSA, self).__init__()
        print(f'in_dim in SSA: {in_dim}')
        assert in_dim % num_heads != 0
        
        self.num_heads = num_heads
        self.head_dim = int(in_dim // num_heads)
        self.scale = self.head_dim ** -0.5

        self.k_proj = nn.Conv2d(in_dim, in_dim, kernel_size=[1, 1], stride=[1, 1], bias=bias)
        self.q_proj = nn.Conv2d(in_dim, in_dim, kernel_size=[1, 1], stride=[1, 1], bias=bias)
        self.v_proj = nn.Conv2d(in_dim, in_dim, kernel_size=[1, 1], stride=[1, 1], bias=bias)
        
        self.attn_drop = nn.Dropout(attn_drop_prob)
        self.proj = nn.Linear(in_dim, in_dim)
        self.proj_drop = nn.Dropout(proj_drop_prob)

    def forward(self, x):
        B, T, N, D = x.shape # batch_size, num_step, num_node, dim

        W_k = self.k_proj(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) \
                  .reshape(B, N, T, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        W_q = self.q_proj(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) \
                  .reshape(B, N, T, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        W_v = self.v_proj(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) \
                  .reshape(B, N, T, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        
        attn = (W_q @ W_k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn_scores = (attn @ W_v).transpose(2, 3).reshape(B, N, T, D)

        x = self.proj(attn_scores)
        x = self.proj_drop(x)
        return x
    
class HTSA(nn.Module):
    def __init__(self, in_dim, num_heads, bias=False, 
                 local_interval=60, daily_interval=1440, weekly_interval=10080,
                 attn_drop_prob=0., proj_drop_prob=0., device='cpu'):
        super(HTSA, self).__init__()
        assert in_dim % num_heads != 0

        self.device = device
        self.num_heads = num_heads
        self.head_dim = int(in_dim // num_heads)
        self.scale = self.head_dim ** -0.5
        self.local_interval = local_interval
        self.daily_interval = daily_interval
        self.weekly_interval = weekly_interval

        self.k_proj = nn.Conv2d(in_dim, in_dim, kernel_size=[1, 1], stride=[1, 1], bias=bias)
        self.q_proj = nn.Conv2d(in_dim, in_dim, kernel_size=[1, 1], stride=[1, 1], bias=bias)
        self.v_proj = nn.Conv2d(in_dim, in_dim, kernel_size=[1, 1], stride=[1, 1], bias=bias)
        
        self.attn_drop = nn.Dropout(attn_drop_prob)
        self.proj = nn.Linear(in_dim, in_dim)
        self.proj_drop = nn.Dropout(proj_drop_prob)

    def _get_mask(self, T, mode):
        mask = torch.zeros(T, T,  device=self.device)
        if mode == 'local':
            for i in range(T):
                start = max(0, i - self.recent_window + 1)
                mask[i, start:i+1] = 1
        elif mode == 'daily':
            for i in range(T):
                idxs = [i - j*self.daily_interval for j in range((i // self.daily_interval)+1) if i - j*self.daily_interval >=0]
                mask[i, idxs] = 1
        elif mode == 'weekly':
            for i in range(T):
                idxs = [i - j*self.weekly_interval for j in range((i // self.weekly_interval)+1) if i - j*self.weekly_interval >=0]
                mask[i, idxs] = 1
        return mask
    
    def forward(self, x):
        B, T, N, D = x.shape # batch_size, num_step, num_node, dim

        W_k = self.k_proj(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1) \
                  .reshape(B, N, T, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        W_q = self.q_proj(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1) \
                  .reshape(B, N, T, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        W_v = self.v_proj(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1) \
                  .reshape(B, N, T, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        
        attns = []
        for mode in ['local', 'daily', 'weekly']:
            mask = self._get_mask(T, mode)
            mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            attn = (W_q @ W_k.transpose(-2, -1)) * self.scale
            
            attn_scores = (attn @ W_v).transpose(2, 3).reshape(B, N, T, D).transpose(1, 2)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            attns.append(attn)

        x = torch.cat(attns, dim=-1) # B,N,num_head,T,head_dim*3
        x = x.permute(0, 3, 1, 2, 4).reshape(B, T, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class SHTFusion(nn.Module):
    '''
    SA: spatial_attn
    HTA: hierarchical_temporal_attn
    '''
    def __init__(self, in_dim):
        super(SHTFusion).__init__()
        self.proj = nn.Conv2d(in_dim, in_dim, kernel_size=[1, 1], stride=[1, 1])

    def forward(self, SA, HTA):
        z = torch.sigmoid(torch.add(SA, HTA))
        x = torch.add(torch.mul(z, SA), torch.mul(1 - z, HTA))
        x = self.proj(x)
        return x
    
class SHTBlock(nn.Module):
    def __init__(self, in_dim, dropout):
        super(SHTBlock).__init__()
        self.spatial_attn = SSA(in_dim=in_dim, num_heads=8)
        self.temporal_attn = HTSA(in_dim=in_dim, num_heads=8)
        self.fusion = SHTFusion()
        self.mlp = nn.Sequential(*[
                                    nn.Linear(in_dim, in_dim * 4),
                                    nn.GELU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(in_dim * 4, in_dim),
                                    nn.Dropout(dropout)
                                ])
        # self.ln1 = nn.LayerNorm(in_dim)
        self.ln = nn.LayerNorm(in_dim)

    def forward(self, x):
        # residual = x
        SA = self.spatial_attn(x)
        HTA = self.temporal_attn(x)
        fusion = self.fusion(SA, HTA)

        # x = residual + fusion
        # x = self.ln1(x)
        x = self.mlp(fusion)
        x = self.ln(x)
        return x
    
class SHTA(nn.Module):
    def __init__(self, args, device, in_dim, num_block=3):
        super(SHTA, self).__init__()
        self.num_nodes = args.num_nodes
        self.out_dim = args.out_dim
        self.dropout = args.dropout
        self.channels = args.channels
        self.horizon = args.horizon
        self.start_fc = nn.Linear(in_features=in_dim, out_features=self.channels)
        self.memory_size = args.memory_size

        self.layers = nn.ModuleList(
            [
                SHTBlock(in_dim=self.channels, dropout=self.dropout) for i in range(num_block)
            ])

        self.skip_layers = nn.ModuleList([
            nn.Linear(in_features=12 * self.channels, out_features=256),
            nn.Linear(in_features=3 * self.channels, out_features=256),
            nn.Linear(in_features=1 *self.channels, out_features=256),
        ])

        self.proj = nn.Sequential(*[
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.horizon * self.out_dim)])

    def forward(self, x):
        x = self.start_fc(x)
        batch_size = x.size(0)
        skip = 0

        for layer, skip_layer in zip(self.layers, self.skip_layers):
            x = layer(x)
            skip_inp = x.transpose(2, 1).reshape(batch_size, self.num_nodes, -1)
            skip = skip + skip_layer(skip_inp)

        x = torch.relu(skip)
        out = self.proj(x)
        if self.out_dim == 1:
            out = out.transpose(2, 1).unsqueeze(-1)
        else:
            out = out.unsqueeze(-1).reshape(batch_size, self.num_nodes, self.horizon, -1).transpose(2, 1)

        return out




