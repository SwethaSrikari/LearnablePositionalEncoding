# Exploring different Positional Encoding methods

### 1. The sinusoidal positional encoding used in the Transformer Model from the paper [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)

![spe](/images/spe_formula.png)

```python
class SinusoidalPositionalEncoding(nn.Module):

  def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
    super().__init__()
    self.dropout = nn.Dropout(p=dropout)
  
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, 1, d_model)
    pe[:, 0, 0::2] = torch.sin(position * div_term)
    pe[:, 0, 1::2] = torch.cos(position * div_term)
    self.register_buffer('pe', pe)
  
  def forward(self, x: Tensor) -> Tensor:
    """
    Arguments:
        x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
    """
    x = x + self.pe[:x.size(0)]
    return self.dropout(x)
```

```python
class TransformerSPE(nn.Module):

  def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
         nlayers: int, dropout: float = 0.5):
    super().__init__()
    self.model_type = 'Transformer_SPE' # changes with encoding
    self.pos_encoder = SinusoidalPositionalEncoding(d_model, dropout) # changes with encoding
    encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
    self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
    self.embedding = nn.Embedding(ntoken, d_model)
    self.d_model = d_model
    self.linear = nn.Linear(d_model, ntoken)
  
    self.init_weights()
  
  def init_weights(self) -> None:
    initrange = 0.1
    self.embedding.weight.data.uniform_(-initrange, initrange)
    self.linear.bias.data.zero_()
    self.linear.weight.data.uniform_(-initrange, initrange)
  
  def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
    """
    Arguments:
        src: Tensor, shape ``[seq_len, batch_size]``
        src_mask: Tensor, shape ``[seq_len, seq_len]``
  
    Returns:
        output Tensor of shape ``[seq_len, batch_size, ntoken]``
    """
    src = self.embedding(src) * math.sqrt(self.d_model)
    src = self.pos_encoder(src) # changes with encoding
    output = self.transformer_encoder(src, src_mask)
    output = self.linear(output)
    return output
```

### The Learnable Sinusoidal Positional Encoding (LSPE) from the paper [A Simple yet Effective Learnable Positional Encoding Method for Improving Document Transformer Model](https://aclanthology.org/2022.findings-aacl.42.pdf)

![lspe](/images/lspe_formula.png)

```python
class LearnableSPE(nn.Module):

  def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
    super().__init__()
    self.dropout = nn.Dropout(p=dropout)
  
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, 1, d_model)
    pe[:, 0, 0::2] = torch.sin(position * div_term)
    pe[:, 0, 1::2] = torch.cos(position * div_term)
    self.register_buffer('pe', pe)
  
  def forward(self, x: Tensor) -> Tensor:
    """
    Arguments:
        x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
    """
    x = x + self.pe[:x.size(0)]
    return self.dropout(x)

class PositionwiseFeedForward(nn.Module):

  def __init__(self, d_model, hidden, drop_prob=0.1):
    super(PositionwiseFeedForward, self).__init__()
    self.linear1 = nn.Linear(d_model, hidden)
    self.linear2 = nn.Linear(hidden, d_model)
    self.sigmoid = nn.Sigmoid()
    self.dropout = nn.Dropout(p=drop_prob)
  
  def forward(self, x):
    x = self.linear1(x)
    x = self.sigmoid(x)
    x = self.dropout(x)
    x = self.linear2(x)
    return x
```

```python
class TransformerLSPE(nn.Module):

  def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
         nlayers: int, dropout: float = 0.5):
    super().__init__()
    self.model_type = 'Transformer_LSPE' # changes with encoding
    self.pos_encoder = LearnableSPE(d_model, dropout) # changes with encoding
    self.ffn = PositionwiseFeedForward(d_model, d_hid) # changes with encoding
    encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
    self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
    self.embedding = nn.Embedding(ntoken, d_model)
    self.d_model = d_model
    self.linear = nn.Linear(d_model, ntoken)
    self.encodings = None
  
    self.init_weights()
  
  def init_weights(self) -> None:
    initrange = 0.1
    self.embedding.weight.data.uniform_(-initrange, initrange)
    self.linear.bias.data.zero_()
    self.linear.weight.data.uniform_(-initrange, initrange)
  
  def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
    """
    Arguments:
        src: Tensor, shape ``[seq_len, batch_size]``
        src_mask: Tensor, shape ``[seq_len, seq_len]``
  
    Returns:
        output Tensor of shape ``[seq_len, batch_size, ntoken]``
    """
    src = self.embedding(src) * math.sqrt(self.d_model)
    src = self.pos_encoder(src) # changes with encoding
    src = self.ffn(src) # changes with encoding
    self.encodings = src
    output = self.transformer_encoder(src, src_mask)
    output = self.linear(output)
    return output
```
