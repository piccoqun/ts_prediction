# reference 1: https://github.com/pytorch/examples/blob/master/time_sequence_prediction/train.py
# reference 2: https://github.com/jcjohnson/pytorch-examples

import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.set_default_tensor_type('torch.DoubleTensor')

# D_in is input dimension;
# H is hidden dimension
seq_len, batch_size, D_in, H = 200, 50, 5, 3

# Create random Tensors to hold inputs and outputs.
x = torch.randn(batch_size, seq_len, D_in) # (50, 200, 5)
y = torch.randn(batch_size, seq_len)

class Sequence(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(input_size, hidden_size)
        self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)
        self.h = hidden_size

    def forward(self, input, future=0):
        outputs = []
        h_t = torch.zeros(input.size(0), self.h)  # torch.Size([batch, hidden])
        c_t = torch.zeros(input.size(0), self.h)
        h_t2 = torch.zeros(input.size(0), self.h)
        c_t2 = torch.zeros(input.size(0), self.h)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstm1(input_t.view(input_t.size(0), input_t.size(2)),(h_t, c_t))  # input_t reshaped to [batch, input]
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):  # if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


model = Sequence(D_in, H)
criterion = nn.MSELoss()

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algoriths. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss_ls = []
for t in range(500):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x)

    # Compute and print loss.
    loss = criterion(y_pred, y)
    loss_ls.append(loss)

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the Tensors it will update (which are the learnable weights
    # of the model)
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its parameters
    optimizer.step()

plt.plot(loss_ls)
plt.savefig('test/test_loss.png')
plt.show()

## device = gpu?