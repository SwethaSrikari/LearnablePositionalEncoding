import math
import torch
from torch import nn, Tensor

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
		self.pe_out = self.ffn(self.pe)
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