import argparse
import math
import time
import os
from tempfile import TemporaryDirectory
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

from models.transformer.transformer_lspe import TransformerLSPE
from models.transformer.transformer_spe import TransformerSPE
from data import get_vocab, get_data
from utils.get_batch import get_batch


# Initiate model instance
vocab, _ = get_vocab()
ntokens = len(vocab)  # size of vocabulary
emsize = 200  # embedding dimension
d_hid = 200  # dimension of the feedforward network model in ``nn.TransformerEncoder``
nlayers = 2  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
nhead = 2  # number of heads in ``nn.MultiheadAttention``
dropout = 0.2  # dropout probability

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data
train_data, val_data = get_data()

# bptt
bptt = 35

# Training hyperparameters
criterion = nn.CrossEntropyLoss()
lr = 5.0  # learning rate

def train(model: nn.Module, optimizer) -> None:
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 200
    start_time = time.time()

    num_batches = len(train_data) // bptt
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        output = model(data)
        output_flat = output.view(-1, ntokens)
        loss = criterion(output_flat, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()

def evaluate(model: nn.Module, eval_data: Tensor) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, i)
            seq_len = data.size(0)
            output = model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += seq_len * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)


def run(epochs=3, model='Transformer_SPE'):
	if model == 'Transformer_SPE':
		model = TransformerSPE(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)
	elif model == 'Transformer_LSPE':
		model = TransformerLSPE(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)
	else:
		print('No such model')
		exit()

	# Training hyperparameters
	optimizer = torch.optim.SGD(model.parameters(), lr=lr)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

	best_val_loss = float('inf')

	for epoch in range(1, epochs + 1):
		epoch_start_time = time.time()
		train(model, optimizer)
		val_loss = evaluate(model, val_data)
		val_ppl = math.exp(val_loss)
		elapsed = time.time() - epoch_start_time
		print('-' * 89)
		print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
		    f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
		print('-' * 89)

		if val_loss < best_val_loss:
			best_val_loss = val_loss
			# torch.save(model.state_dict(), best_model_params_path)

		scheduler.step()
    # model.load_state_dict(torch.load(best_model_params_path)) # load best model states

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--epochs', type=int, default=3, help='Number of epochs to train')
	parser.add_argument('--model', type=str, default='Transformer_SPE',
						help='Type of positional enccoding to use')
	args = parser.parse_args()

	run(args.epochs, args.model)