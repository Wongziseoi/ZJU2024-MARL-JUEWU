# A Simple RNN Task: 利用RNN的二元分类网络区分不同函数
# cmd> pip install torch numpy matplotlib 

from typing import Iterator
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

DATA_SIZE = 500
category_size = 3

sine_data_propotion = np.random.uniform(0.3, 0.7)
quadratic_data_propotion = np.random.uniform(0.3, 0.7)
sigmoid_data_propotion = np.random.uniform(0.3, 0.7)

sine_data_size = int(DATA_SIZE * sine_data_propotion / (sine_data_propotion + quadratic_data_propotion + sigmoid_data_propotion))
quadratic_data_size = int(DATA_SIZE * quadratic_data_propotion / (sine_data_propotion + quadratic_data_propotion + sigmoid_data_propotion))
sigmoid_data_size = DATA_SIZE - sine_data_size - quadratic_data_size

steps = np.arange(0, 10, 0.5)

# generate sine-like function samples
sine_init = np.random.uniform(-3, 3, (sine_data_size, 2))  # randomize a and b for sin(ax+b)
sine_data = np.sin(sine_init[:, :1] * steps + sine_init[:, 1:])

# generate linear-like function samples
quadratic_init = np.random.uniform(-3, 3, (quadratic_data_size, 2)) # randomize a and b for ax^2+bx
quadratic_data = quadratic_init[:, :1] * steps ** 2 + quadratic_init[:, 1:]


# generate sigmoid-like function samples
sigmoid_init = np.random.uniform(-3, 3, (sigmoid_data_size, 2)) # randomize a and b for 1/(1+e^(-ax+b))
sigmoid_data = 1 / (1 + np.exp(0 - sigmoid_init[:, :1] * steps + sigmoid_init[:, 1:]))

fig, axs = plt.subplots(1, 3)
axs[0].plot(sine_data[0])
axs[1].plot(sigmoid_data[1])
axs[2].plot(quadratic_data[2])
plt.show()

# mix data
sine_data = np.concatenate((sine_data, np.broadcast_to(np.array([1,0,0]), (sine_data_size, 3))), axis=1)
quadratic_data = np.concatenate((quadratic_data, np.broadcast_to(np.array([0,1,0]), (quadratic_data_size, 3))), axis=1)
sigmoid_data = np.concatenate((sigmoid_data, np.broadcast_to(np.array([0,0,1]), (sigmoid_data_size, 3))), axis=1)
data = np.concatenate((sine_data, sigmoid_data, quadratic_data), axis=0)
data = torch.Tensor(data)

# split two datasets
from torch.utils.data import random_split
train_set, test_set = random_split(data, [0.8, 0.2])

class SimpleClassificationRNN(nn.Module):
    def __init__(self, hidden_size):
        super(SimpleClassificationRNN, self).__init__()
        # self.rnn = nn.RNN(input_size=1,
        #                   hidden_size=hidden_size,
        #                   batch_first=True,
        #                   num_layers=4,
        #                   bidirectional=True)
        self.rnn = nn.LSTM(input_size=1,
                            hidden_size=hidden_size,
                            batch_first=True,
                            num_layers=1)
        self.linear = nn.Linear(hidden_size, 3)

    def forward(self, seq, hc=None, is_train=True):
        tmp, (hc, cc) = self.rnn(seq, hc)
        x = self.linear(hc[-1, ... ,:])
        if is_train:
            out = torch.sigmoid(x)
            out = out / out.sum(dim=1, keepdim=True)
            return out, hc, cc  
        else:
            return x, hc, cc
    
    def predict(self, seq):
        with torch.no_grad():
            output, _, _ = self.forward(seq)
        return output.argmax()
    
hidden_size = 30
learning_rate = 0.01

model = SimpleClassificationRNN(hidden_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), learning_rate)

# you can also test other optimizers and learning_rate settings ...
def cal_accuracy(preds, true_values):
    preds = preds.argmax(dim=1)
    acc = torch.sum(preds == true_values) / preds.shape[0]
    return acc

# you can also implement other metrics like F1 ...
epochs = 100
loss_log = []
for epoch in range(epochs):
    optimizer.zero_grad()
    output, _, _ = model(train_set[:][:, :-category_size, np.newaxis])
    loss = criterion(output.view(-1, 3), train_set[:][:, -category_size:])
    acc = cal_accuracy(output.view(-1, 3), train_set[:][:, -category_size:].argmax(dim=1))
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print("Epoch {}: loss {} acc {}".format(epoch, loss.item(), acc))
    if epoch == epochs - 1:
        print("Preds: {}, True: {}".format(output.argmax(dim=1), train_set[:][:, -category_size:].argmax(dim=1)))
        print(output.view(-1, 3))

# you can also implement early stopping here ...
output, _, _ = model(test_set[:][:, :category_size, np.newaxis])
loss = criterion(output.view(-1, 3), test_set[:][:, -category_size:])
acc = cal_accuracy(output.view(-1, 3), test_set[:][:, -category_size:].argmax(dim=1))

# print("Preds: {}, True: {}".format(output.argmax(dim=1), test_set[:][:, -category_size:].argmax(dim=1)))

print("Test set: loss {} acc {}".format(loss.item(), acc))
