# reference 1: https://github.com/pytorch/examples/blob/master/time_sequence_prediction/train.py
# reference 2: https://github.com/jcjohnson/pytorch-examples

# todo: add dropout and gpu

import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class LSTM:
    def __init__(self, input_size, hidden_size):
        torch.set_default_tensor_type('torch.FloatTensor')
        self.model = Sequence(input_size, hidden_size)

    def train(self, train_gen, epoch):
        criterion = nn.MSELoss()
        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use Adam; the optim package contains many other
        # optimization algoriths. The first argument to the Adam constructor tells the
        # optimizer which Tensors it should update.
        learning_rate = 1e-4
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        print('epochs: ', epoch)
        for i in range(epoch):
            for step, batch in enumerate(train_gen):
                # Before the backward pass, use the optimizer object to zero all of the
                # gradients for the Tensors it will update (which are the learnable weights
                # of the model)
                optimizer.zero_grad()
                batch_x = torch.tensor(batch[0], dtype=torch.float)
                batch_y = torch.tensor(batch[1], dtype=torch.float)
                # Forward pass: compute predicted y by passing x to the model.
                y_pred = self.model(batch_x)
                # Compute and print loss.
                loss = criterion(y_pred, batch_y)
                print('epoch',i, ' step', step, ' loss',loss)
                # Backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # Calling the step function on an Optimizer makes an update to its parameters
                optimizer.step()


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
            h_t, c_t = self.lstm1(input_t.squeeze(1),(h_t, c_t))  # input_t reshaped to [batch, input]
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):  # to predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1)[:, -1] # we only care about the last value in the sequence
        return outputs


## device = gpu?