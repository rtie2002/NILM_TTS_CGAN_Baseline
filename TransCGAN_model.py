import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch import Tensor 
import math 
import numpy as np

from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Generator(nn.Module):
    def __init__(self, seq_len=150, channels=1, num_classes=9, latent_dim=100, data_embed_dim=10, 
                label_embed_dim=10, depth=3, num_heads=5, 
                forward_drop_rate=0.5, attn_drop_rate=0.5, time_dim=8):
        super(Generator, self).__init__()
        self.seq_len = seq_len
        self.channels = channels
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.data_embed_dim = data_embed_dim
        self.label_embed_dim = label_embed_dim
        self.time_dim = time_dim
        self.depth = depth
        self.num_heads = num_heads
        
        # 🚀 NEW: Point-wise input projection
        # Input to Transformer will be: [z (expanded), c (expanded), time (point-wise)]
        self.input_dim = self.latent_dim + self.label_embed_dim + self.time_dim
        self.input_proj = nn.Linear(self.input_dim, self.data_embed_dim)
        
        # 🚀 NEW: Add Positional Encoding
        self.pos_encoder = PositionalEncoding(self.data_embed_dim, max_len=self.seq_len)
        
        self.label_embedding = nn.Embedding(self.num_classes, self.label_embed_dim) 
        
        self.blocks = Gen_TransformerEncoder(
                 depth=self.depth,
                 emb_size = self.data_embed_dim,
                 num_heads = self.num_heads,
                 drop_p = attn_drop_rate,
                 forward_drop_p=forward_drop_rate
                )

        # 🚀 CRITICAL FIX: Use Conv1d with large kernel to enforce smoothness
        # Previous 1x1 Conv caused independent "spikes". 
        # Kernel=9 acts as a smoothing filter, mixing neighbors.
        self.deconv = nn.Sequential(
            nn.Conv1d(self.data_embed_dim, self.data_embed_dim, kernel_size=9, padding=4),
            nn.LeakyReLU(0.2),
            nn.Conv1d(self.data_embed_dim, self.channels, kernel_size=1)
        )
        
    def forward(self, z, labels, time_features):
        batch_size = z.shape[0]
        
        # 1. Expand Latent Noise: (B, Latent) -> (B, Seq, Latent)
        z_expanded = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        
        # 2. Expand Label Embedding: (B, LabelDim) -> (B, Seq, LabelDim)
        c = self.label_embedding(labels)
        c_expanded = c.unsqueeze(1).repeat(1, self.seq_len, 1)
        
        # 3. Prepare Time Features: (B, 8, 1, 512) -> (B, 512, 8)
        # We need to correctly reshape and permute to match sequence dim
        t_seq = time_features.squeeze(2).permute(0, 2, 1) 
        
        # 4. Concatenate Point-wise: (B, Seq, Latent + Label + Time)
        x = torch.cat([z_expanded, c_expanded, t_seq], dim=2)
        
        # 5. Project to Embedding Dim
        x = self.input_proj(x) # (B, Seq, Embed)
        
        # 🚀 Apply Positional Encoding
        x = x.permute(1, 0, 2) # (Seq, Batch, Feat)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2) # Back to (Batch, Seq, Feat)
        
        x = self.blocks(x)
        # x is (Batch, Seq, Embed)
        
        # 🚀 Transpose for Conv1d: (Batch, Embed, Seq)
        x = x.permute(0, 2, 1)
        
        output = self.deconv(x)
        
        # Reshape to (Batch, 1, 1, Seq) to match original expected output format
        output = output.unsqueeze(2) 
        
        return torch.sigmoid(output)


class Gen_TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=5,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))
        
class Gen_TransformerEncoder(nn.Sequential):
    def __init__(self, depth=8, **kwargs):
        super().__init__(*[Gen_TransformerEncoderBlock(**kwargs) for _ in range(depth)]) 
        

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

    
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x
    
    
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

     

class Dis_TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size=100,
                 num_heads=5,
                 drop_p=0.,
                 forward_expansion=4,
                 forward_drop_p=0.):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class Dis_TransformerEncoder(nn.Sequential):
    def __init__(self, depth=8, **kwargs):
        super().__init__(*[Dis_TransformerEncoderBlock(**kwargs) for _ in range(depth)])
        
        
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size=100, adv_classes=2, cls_classes=10):
        super().__init__()
        self.adv_head = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, adv_classes)
        )
        self.cls_head = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, cls_classes)
        )

    def forward(self, x):
        out_adv = self.adv_head(x)
        out_cls = self.cls_head(x)
        return out_adv, out_cls

    
class PatchEmbedding_Linear(nn.Module):
    def __init__(self, in_channels = 21, patch_size = 16, emb_size = 100, seq_length = 1024):
        super().__init__()
        self.projection = nn.Sequential(
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)',s1 = 1, s2 = patch_size),
            nn.Linear(patch_size*in_channels, emb_size)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn((seq_length // patch_size) + 1, emb_size))


    def forward(self, x:Tensor) ->Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.positions
        return x    

        
class Discriminator(nn.Module):
    def __init__(self, in_channels=9, seq_length=512, **kwargs):
        super(Discriminator, self).__init__()
        
        # 🚀 TCN Discriminator: Scrutinizes every detail using Convolution
        # No more patches, no more hiding spikes!
        
        self.main = nn.Sequential(
            # Input: (B, 9, 512)
            nn.Conv1d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (B, 64, 256)
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (B, 128, 128)
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (B, 256, 64)
            nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (B, 512, 32)
            nn.Conv1d(512, 1, kernel_size=32, stride=1, padding=0),
            # Output: (B, 1, 1) -> Scalar Score
        )
        
    def forward(self, power, time_features):
        # Flatten input: Power (B, 1, 1, 512) -> (B, 1, 512)
        p = power.squeeze(2) 
        # Time (B, 8, 1, 512) -> (B, 8, 512)
        t = time_features.squeeze(2)
        
        # Concatenate: (B, 9, 512)
        x = torch.cat([p, t], dim=1)
        
        output = self.main(x)
        
        # Return score (B, 1) and None (no class logits needed for WGAN-GP)
        return output.view(-1, 1), None