import torch
import torch.nn as nn

seq_len, batch_size, D_in, H = 200, 50, 5, 3

# Create random Tensors to hold inputs and outputs.
x = torch.randn(batch_size, seq_len, D_in) # (50, 200, 5)
y = torch.randn(batch_size, 1)

from predictionmodel.lstm_pytorch import model
model = model(D_in, H)
model.train(x,y)