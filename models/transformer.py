import math
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from models.positional_encoding.LSPE import LearnableSPE
from models.positional_encoding.SPE import SinusoidalPositionalEncoding

class Transformer(nn.Module):

	def __init__(self, encoding, ntoken: int, d_model: int, nhead: int, d_hid: int,
				 nlayers: int, dropout: float = 0.5):
		super().__init__()
		self.model_type = 'Transformer_LSPE'
		self.pos_encoder = encoding
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
		src = self.pos_encoder(src)
		output = self.transformer_encoder(src, src_mask)
		output = self.linear(output)
		return output