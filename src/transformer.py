import numpy as np
import torch
import torch.nn as nn
import math, copy, time


## ----------------------------------------------------------------------------
## CORE MODULES
## ----------------------------------------------------------------------------

class ScaledDotProductAttention(nn.Module):

    def __init__(self, heads, key_dim, dropout=0.1):
        super().__init__()
        
        self.heads = heads
        self.multiplier = math.sqrt(key_dim)
        # dropout
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, q, k, v, mask=None):
        '''
        computes scaled dot product attention
        
        attention = Softmax((Q K^T) / sqrt(d_k)) V
        '''        
        # calculate score 
        score = torch.matmul(q, k.transpose(-2, -1))
        score = score / self.multiplier  
        # mask score if provided
        if mask is not None:
            mask = mask.unsqueeze(1)  # for the 'head' dim
            score = score.masked_fill(mask, -np.inf)
        
        score = torch.softmax(score, dim=-1)    
        score = self.dropout(score)
        attn = torch.matmul(score, v)    
        return attn


## ----------------------------------------------------------------------------
## SUBLAYERS
## ----------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    
    def __init__(self, heads, model_dim, key_dim, value_dim, dropout=0.1):
        '''
        Multi-head attention for the transformer
        
        Parameters:
        -----------
        heads: number of heads to use
        model_dim: model dimension i.e., input dimensions for query, key and value
        key_dim: key dimension for query & key
        value_dim: value dimension for value
        '''
        super().__init__()
        
        self.heads = heads
        self.d_model = model_dim
        self.d_k = key_dim
        self.d_v = value_dim
        
        # initialize weights & biases
        self.Wq = nn.Linear(self.d_model, self.heads*self.d_k)
        self.Wk = nn.Linear(self.d_model, self.heads*self.d_k)
        self.Wv = nn.Linear(self.d_model, self.heads*self.d_v)
        
        # initialize self attention
        self.attn = ScaledDotProductAttention(heads = self.heads, key_dim=self.d_k)
        
        # final fc layer combining all attention heads
        self.Wout = nn.Linear(self.heads*self.d_v, self.d_model)
        
        self.dropout = nn.Dropout(dropout)
        # layer norm for residual connection
        self.norm = nn.LayerNorm(self.d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        '''
        Runs multi-head attention on the given inputs
        
        Parameters:
        -----------
        q: Query - Tensor [batch, seq_len, d_model]
        k: Key   - Tensor [batch, seq_len, d_model]
        v: Value - Tensor [batch, seq_len, d_model]
        mask: masking tensor for the score
            Tensor [seq_len, seq_len]
            
        Returns:
        --------
        attention: Combined multi-head attention 
            Tensor of shape [seq_len, batch, d_model]
        '''
        
        batch = q.shape[0]
        residual = q
        
        # linear operation on inputs - output shape: [batch, seq_len, heads*dim]
        # reshape to [batch, heads, seq_len, dim] for computation efficiency
        q = self.Wq(q).view(batch, -1, self.heads, self.d_k).transpose(1, 2)
        k = self.Wk(k).view(batch, -1, self.heads, self.d_k).transpose(1, 2)
        v = self.Wv(v).view(batch, -1, self.heads, self.d_v).transpose(1, 2)
        
        # calling self attention - output shape: [batch, heads, seq_len, d_v]
        attn = self.attn(q, k, v, mask)
        # reshape into [batch, seq_len, heads*d_v]
        attn = attn.transpose(1,2).contiguous().view(batch, -1, self.heads*self.d_v)
        
        # combine multihead attention into one
        attn = self.Wout(attn)
        # add and normalize
        attn = self.dropout(attn)
        attn = self.norm(residual + attn)
        
        return attn


class PositionWiseFeedForward(nn.Module):
    
    def __init__(self, model_dim, ff_dim, dropout=0.1):
        ''' 
        Position wise feed forward network, through which all individual elements are passed 
        
        Parameters:
        -----------
        model_dim: model dimension i.e., input dimension of the tensor
        ff_dim: hidden dimension for the feed forward network
        '''
        super().__init__()
        
        self.w1 = nn.Linear(model_dim, ff_dim)
        self.w2 = nn.Linear(ff_dim, model_dim)
        
        self.dropout = nn.Dropout(dropout)
        # layer norm for residual connection
        self.norm = nn.LayerNorm(model_dim, eps=1e-6)
        
    def forward(self, x):
        ''' 
        Runs a feed forward network on each input encoding
        Input and output shape - [batch, seq_len, d_model]
        '''
        out = torch.relu(self.w1(x))    
        out = torch.relu(self.w2(out))

        out = self.dropout(out)
        # add and normalize
        out = self.norm(out + x)
        return out


## ----------------------------------------------------------------------------
## MODULES - ENCODER & DECODER
## ----------------------------------------------------------------------------

class EncoderCell(nn.Module):
    
    def __init__(self, heads, model_dim, key_dim, value_dim, ff_dim, dropout=0.1):
        '''
        One layer of a transformer encoder with multi-head attention and position wise feed forward
        
        Parameters:
        -----------
        heads: number of heads to use
        model_dim: model dimension i.e., input dimensions for query, key and value
        key_dim: key dimension for query & key
        value_dim: value dimension for value
        ff_dim: hidden dimension for the feed forward network
        '''
        super().__init__()
        
        self.self_attn = MultiHeadAttention(heads, model_dim, key_dim, value_dim, dropout)
        self.pos_ff = PositionWiseFeedForward(model_dim, ff_dim, dropout)
        
    def forward(self, x, mask):
        '''
        Parameters:
        -----------
        x: input tensor 
        mask: masking tensor with same shape as input
        
        Input and output shape - [seq_len, batch, d_model]
        '''    
        # get attention
        attn = self.self_attn(x, x, x, mask)
        # position wise feed forward
        pos = self.pos_ff(attn)
        
        return pos


class DecoderCell(nn.Module):
    
    def __init__(self, heads, model_dim, key_dim, value_dim, ff_dim, dropout=0.1):
        '''
        One layer of a transformer decoder with multi-head attention and position wise feed forward
        
        Parameters:
        -----------
        heads: number of heads to use
        model_dim: model dimension i.e., input dimensions for query, key and value
        key_dim: key dimension for query & key
        value_dim: value dimension for value
        ff_dim: hidden dimension for the feed forward network
        '''
        super().__init__()
        
        self.self_attn = MultiHeadAttention(heads, model_dim, key_dim, value_dim, dropout)
        self.src_attn = MultiHeadAttention(heads, model_dim, key_dim, value_dim, dropout)
        self.pos_ff = PositionWiseFeedForward(model_dim, ff_dim, dropout)
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        '''
        Parameters:
        -----------
        src: tensor from the last encoder layer
        tgt: positionally encoded output tensor
        src_mask: masking tensor for encoded output with same shape
        tgt_mask: masking tensor for input with same shape
        
        Input and output shape - [seq_len, batch, d_model]
        '''    
        ## get self attention
        tgt_attn = self.self_attn(tgt, tgt, tgt, tgt_mask)
        ## get src attention (from encoder)
        src_attn = self.src_attn(tgt_attn, src, src, src_mask)
        ## position wise feed forward
        pos = self.pos_ff(src_attn)
        return pos


class Encoder(nn.Module):
    
    def __init__(self, N, heads, model_dim, key_dim, value_dim, ff_dim, dropout=0.1):
        '''
        Encoder module of a transformer with N encoder layers
        
        Parameters:
        -----------
        N: number of encoder layers
        heads: number of heads to use
        model_dim: model dimension i.e., input dimensions for query, key and value
        key_dim: key dimension for query & key
        value_dim: value dimension for value
        ff_dim: hidden dimension for the feed forward network
        '''
        super().__init__()
        
        self.enc_stack = nn.ModuleList([
                EncoderCell(heads, model_dim, key_dim, value_dim, ff_dim, dropout)
                for _ in range(N)])
        
        # self.norm = nn.LayerNorm(model_dim, eps=1e-6)
        
    def forward(self, x, mask):
        '''
        Parameters:
        -----------
        x: input tensor 
        mask: masking tensor with same shape as input
        
        Input and output shape - [seq_len, batch, d_model]
        '''
        # get attention
        for enc in self.enc_stack:
            x = enc(x, mask)
        # x = self.norm(x)
        return x


class Decoder(nn.Module):
    
    def __init__(self, N, heads, model_dim, key_dim, value_dim, ff_dim, dropout=0.1):
        '''
        Decoder module of a transformer with N decoder layers
        
        Parameters:
        -----------
        N: number of decoder layers
        heads: number of heads to use
        model_dim: model dimension i.e., input dimensions for query, key and value
        key_dim: key dimension for query & key
        value_dim: value dimension for value
        ff_dim: hidden dimension for the feed forward network
        '''
        super().__init__()
        
        self.dec_stack = nn.ModuleList([
                DecoderCell(heads, model_dim, key_dim, value_dim, ff_dim, dropout)
                for _ in range(N)])
        
        # self.norm = nn.LayerNorm(model_dim, eps=1e-6)
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        '''
        Parameters:
        -----------
        x: input tensor 
        mask: masking tensor with same shape as input
        
        Input and output shape - [seq_len, batch, d_model]
        '''
        # get attention
        for dec in self.dec_stack:
            tgt = dec(src, tgt, src_mask, tgt_mask)
        # tgt = self.norm(tgt)
        return tgt


## ----------------------------------------------------------------------------
## SUPPLEMENTARY REQUIRED MODULES
## ----------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len=10000):
        ''' 
        Adds sequential information to a sequence
        
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model)) 
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model)) 
        
        Parameters:
        -----------
        model_dim: model dimension i.e., input dimension of the tensor
        '''
        super().__init__()
        
        self.d_model = model_dim
        self.max_len = max_len
        
        self.pe = torch.zeros(max_len, self.d_model)#.float()
        
        position = torch.arange(0, max_len).unsqueeze(1).float()
        i = torch.arange(0, self.d_model, 2).float()
        div = 10000 ** (i / self.d_model)
        # add sin & cos signals
        self.pe[:,0::2] = torch.sin(position / div)
        self.pe[:,1::2] = torch.cos(position / div)
        self.pe.requires_grad = False
        
    def forward(self, x):
        '''
        Add positional encoding to the given input - per sequence
        Input & output shape - [batch, seq_len, d_model]
        '''
        x = x + self.pe[:x.shape[1], :].to(x.device)
        return x


## Masks method for decoder
def get_subsequent_mask(seq):
    '''
    Generates mask with all positions after current position being masked out
    Returns:
    --------
    a mask of shape [dim, dim] with True in upper triangular matrix
    '''
    dim = seq.shape[1]
    mask = torch.ones(1, dim, dim, dtype=torch.uint8, device=seq.device)
    mask = torch.triu(mask, 1)
    return mask

## Mask for padding
def get_pad_mask(seq_k, seq_q, pad=0):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    # len_q = seq_q.size(1)
    padding_mask = seq_k.eq(pad)
    # padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    padding_mask = padding_mask.unsqueeze(1)  # b x 1 x lk (for any lq)

    return padding_mask


## Mask for padding, if sequence is 3D (padding is a sequence)
def get_pad_mask_n_dim(seq_k, seq_q, pad_pos=0):
    ''' For masking out the padding part of key sequence. '''

    assert seq_k.dim() == 3 and seq_q.dim() == 3
    
    # generate padding vector for position
    pad = torch.zeros(seq_k.shape[2]).to(seq_k.device)
    if pad_pos >= 0:
        pad[pad_pos] = 1

    # Expand to fit the shape of key query attention matrix.
    # len_q = seq_q.size(1)
    padding_mask = torch.all(seq_k.eq(pad), dim=-1)  # b x lq
    padding_mask = padding_mask.unsqueeze(1)#.expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask


## Transformer optimizer
class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()