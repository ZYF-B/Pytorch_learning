import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sequence_len = 64
emb_size = 128
n_head = 8
n_layer = 6
head_size = emb_size//n_head

def attention(query, key, value, mask=None):
    B, T, H = query.shape
    sorces = query @ key.transpose(-1, -2) / (H ** 0.5)
    if mask is not None:
        sorces = sorces.masked_fill(mask==0, float('-inf'))
    sorces = torch.softmax(sorces, dim=-1)
    ouput = sorces @ value
    return ouput

class SelfAttention(nn.Module):

    def __init__(self, emb_size, head_size):
        # emb_size: C, head_size: H
        super().__init__()
        self.key = nn.Linear(emb_size, head_size, bias=False)
        self.query = nn.Linear(emb_size, head_size, bias=False)
        self.value = nn.Linear(emb_size, head_size, bias=False)
        self.dp= nn.Dropout(0.2)

    def forward(self, x, mask=None):
        # x:   (B, T, C)
        # out: (B, T, H)
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        out = attention(q, k, v, mask)
        return self.dp(out)

class MultiHeadSelfAttention(nn.Module):

    def __init__(self, emb_size, head_size):
        super().__init__()
        n_head = emb_size // head_size
        heads = [SelfAttention(emb_size, head_size) for _ in range(n_head)]
        self.heads = nn.ModuleList(heads)
        self.proj = nn.Linear(emb_size, emb_size)
        self.dp = nn.Dropout(0.2)

    def forward(self, x, mask=None):
        # x:   (B, T, C)
        # out: (B, T, C)
        out = torch.concat([h(x, mask) for h in self.heads], dim=-1)  # (B, T, C)
        out = self.dp(self.proj(out))                           # (B, T, C)
        return out
    
class FeedForward(nn.Module):

    def __init__(self, emb_size):
        super().__init__()
        self.ln1 = nn.Linear(emb_size, 4 * emb_size)
        self.ln2 = nn.Linear(4 * emb_size, emb_size)
        self.dp = nn.Dropout(0.2)

    def forward(self, x):
        # x: (B, T, C)
        out = F.gelu(self.ln1(x))     # (B, T, C)
        out = self.dp(self.ln2(out))  # (B, T, C)
        return out

class Encoder_block(nn.Module):
    
    def __init__(self, emb_size, head_size) :
        super().__init__()
        self.l1 = nn.LayerNorm(emb_size)
        self.mha = MultiHeadSelfAttention(emb_size, head_size)
        self.l2 = nn.LayerNorm(emb_size)
        self.ff = FeedForward(emb_size)

    def forward(self, x, mask=None):
        x = x + self.l1(self.mha(x, mask))
        x = x + self.l2(self.ff(x))
        return x

class Encoder(nn.Module):

    # vs : token字典大小
    def __init__(self, vs) :
        super().__init__()
        self.token_emb = nn.Embedding(vs, emb_size)
        self.pos_emb = nn.Embedding(sequence_len, emb_size)
        block = [Encoder_block(emb_size, head_size) for _ in range(n_layer)]
        self.blocks = nn.Sequential(*block)
    
    def forward(self, x, mask=None):
        B, T = x.shape[0]
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        token_embeddings = self.token_emb(x)        
        position_embeddings = self.pos_emb(pos)     
        h = token_embeddings + position_embeddings
        for encoderblk in self.blocks:
            h = encoderblk(h, mask)
        return h
    
class CrossAttention(nn.Module):

    def __init__(self, emb_size, head_size):
        # emb_size: C, head_size: H
        super().__init__()
        self.key = nn.Linear(emb_size, head_size, bias=False)
        self.query = nn.Linear(emb_size, head_size, bias=False)
        self.value = nn.Linear(emb_size, head_size, bias=False)
        self.dp= nn.Dropout(0.2)

    def forward(self, x, x_en, mask=None):
        k = self.key(x_en)
        q = self.query(x)
        v = self.value(x_en)
        out = attention(q, k, v, mask)
        return self.dp(out)

class MultiHeadCrossAttention(nn.Module):

    def __init__(self, emb_size, head_size):
        super().__init__()
        n_head = emb_size // head_size
        heads = [CrossAttention(emb_size, head_size) for _ in range(n_head)]
        self.heads = nn.ModuleList(heads)
        self.proj = nn.Linear(emb_size, emb_size)
        self.dp = nn.Dropout(0.2)

    def forward(self, x, x_en, mask=None):
        # x:   (B, T, C)
        # out: (B, T, C)
        out = torch.concat([h(x, x_en, mask) for h in self.heads], dim=-1)  
        out = self.dp(self.proj(out))                          
        return out


class Decoder_block(nn.Module):

    def __init__(self, emb_size, head_size) :
        super().__init__()
        self.l1 = nn.LayerNorm(emb_size)
        self.mha_self = MultiHeadSelfAttention(emb_size, head_size)
        self.l2 = nn.LayerNorm(emb_size)
        self.mha_cross = MultiHeadCrossAttention(emb_size, head_size)
        self.l3 = nn.LayerNorm(emb_size)
        self.ff = FeedForward(emb_size)
        mask_mat = torch.ones(sequence_len,sequence_len)
        self.mask = torch.tril(mask_mat).unsqueeze(0)

    def forward(self, x, x_en) :
        x = x + self.l1(self.mha_self(x, self.mask))
        x = x + self.l2(self.mha_cross(x, x_en, self.mask))
        x = x + self.l3(self.ff(x))
        return x

class Decoder(nn.modules):

    def __init__(self, vs) :
        super().__init__()
        self.token_emb = nn.Embedding(vs, emb_size)
        self.pos_emb = nn.Embedding(sequence_len, emb_size)
        block = [Decoder_block(emb_size, head_size) for _ in range(n_layer)]
        self.blocks = nn.Sequential(*block)
    
    def forward(self, x, x_en):
        B, T = x.shape[0]
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        token_embeddings = self.token_emb(x)        
        position_embeddings = self.pos_emb(pos)     
        h = token_embeddings + position_embeddings
        for decoderblk in self.blocks:
            h = decoderblk(h, x_en)
        return h
    
class Transformer(nn.Module):

    def __init__(self, vs):
        self.encoder = Encoder(vs)
        self.decoder = Decoder(vs)
    
    def forward(self, x, x_t):
        x_en = self.encoder(x)
        return self.decoder(x, x_en)
