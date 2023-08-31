# Exploring different Positional Encoding methods

### NOTE
This repository focuses only on exploring various positional encoding methods (used to embed sequential order information). The original code has been adopted from this Pytorch Language Modeling [tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html). This tutorial code has been used as the base code for experimenting with different positional encoding methods.

The positional encoding code blocks have been edited and implemented by me.

### 1. The sinusoidal positional encoding used in the Transformer Model from the paper [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)

<img src="/images/spe_formula.png" alt="Image Description" width="700" height="200">
<!-- ![spe](/images/spe_formula.png) -->

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
    x = x + self.pe[:x.size(0)] # positional encodings are fixed
    return self.dropout(x)
```

**Similarity of 1D Sinusoidal position embedding**

<img src="/images/spe_sim.png" alt="Image Description" width="500" height="500">

The given heatmap illustrates how similar each positional embedding is to itself and other positions using cosine similarity. The model employs a positional encoding of size(512, 200), accommodating sequences of up to 512 words. The embedding dimension used for each position is 200. This means that each position in the sequence is represented by a vector of length 200. With the fixed sinusoidal positional embeddings, embeddings of the same position are likely to be highly similar thus showing a diagonal line on the heatmap.

### 2. The Learnable Sinusoidal Positional Encoding (LSPE) from the paper [A Simple yet Effective Learnable Positional Encoding Method for Improving Document Transformer Model](https://aclanthology.org/2022.findings-aacl.42.pdf)

<img src="/images/lspe_formula.png" alt="Image Description" width="500" height="300">
<!-- ![lspe](/images/lspe_formula.png) -->

```python
class LearnableSPE(nn.Module):

  def __init__(self, d_model: int, d_hid: int, dropout: float = 0.1, max_len: int = 512):
    super().__init__()
    self.dropout = nn.Dropout(p=dropout)
    self.ffn = PositionwiseFeedForward(d_model, d_hid)
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
    self.pe_out = self.ffn(self.pe) # Add nn so that encoding becomes trainable
    x = x + self.pe_out[:x.size(0)]
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
**NOTE**
The heatmap below is the result of training the positional encoding for just a few epochs on a very small dataset.

**Similarity of 1D Learnable Sinusoidal position embedding**

Unlike the fixed positional embeddings, these sinusoidal positional embeddings are learned during training. With just a few epochs of training on a small dataset, the embeddings are trying to capture relationships between positions (words) far away from itself.

<img src="/images/lspe_sim.png" alt="Image Description" width="500" height="500">

**Run on Google Colab** - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]([https://colab.research.google.com/github/weiji14/deepbedmap/](https://colab.research.google.com/github/SwethaSrikari/ExploringPositionalEncoding/blob/main/Exploring_positional_encoding.ipynb)https://colab.research.google.com/github/SwethaSrikari/ExploringPositionalEncoding/blob/main/Exploring_positional_encoding.ipynb)

**Run on terminal**

```
python train.py --epochs 10 --encoding 'LSPE'
```

### References

Fixed Sinusoidal Positional Embeddings - [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)

Learnable Sinusoidal Positional Embeddings - [A Simple yet Effective Learnable Positional Encoding Method for Improving Document Transformer Model](https://aclanthology.org/2022.findings-aacl.42.pdf)

Base Transformer code - Pytorch Language Modeling [tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
