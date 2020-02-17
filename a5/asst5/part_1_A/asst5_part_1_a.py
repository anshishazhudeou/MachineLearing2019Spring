# -*- coding: utf-8 -*-

# %matplotlib inline
from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import random

import unicodedata
import string
import math

def findFiles(path): return glob.glob(path)


all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

category_lines = {}
train_data = {}
test_data = {}
all_categories = []


# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    random.shuffle(lines)
    train_data[category] = lines[0:int(math.floor(0.8 * len(lines)))]
    test_data[category] = lines[int(math.floor(0.8 * len(lines))) + 1:]
    category_lines[category] = lines

n_categories = len(all_categories)

import torch


# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)


# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

import torch
import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
  def __init__(self, input_size, hidden_size, output_size, model="linear", n_layers=1):
    super(RNN, self).__init__()
    self.model = model.lower()
    self.hidden_size = hidden_size
    self.n_layers = n_layers


    if self.model == "gru":
      self.rnn = nn.GRU(input_size, hidden_size, n_layers)
      self.decoder = nn.Linear(hidden_size, output_size)
    elif self.model == "lstm":
      self.rnn = nn.LSTM(input_size, hidden_size, n_layers)
      self.decoder = nn.Linear(hidden_size, output_size)
    else:
      self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
      self.i2o = nn.Linear(input_size + hidden_size, output_size)

    self.softmax = nn.LogSoftmax(dim=1)

  def forward(self, input, hidden):
    
    if self.model == 'gru' or self.model == 'lstm':
      output, hidden = self.rnn(input.view(1, 1, -1), hidden)
      output = self.decoder(output.view(1, -1))
      output = self.softmax(output)
      return output, hidden
    else:
      combined = torch.cat((input, hidden), 1)
      hidden = self.i2h(combined)
      output = self.i2o(combined)
      output = self.softmax(output)
      return output, hidden

  def initHidden(self, batch_size):
    if self.model == "lstm":
      return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                    Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))
    elif self.model == "gru":
      return Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
    else:
      return torch.zeros(1, self.hidden_size)

import random
import time
import math

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(train_data[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor  

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# Just return an output given a line
def evaluate(line_tensor):
    hidden = rnn.initHidden(1)
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    return output

criterion = nn.NLLLoss()

learning_rate = 0.005  # If you set this too high, it might explode. If too low, it might not learn


def train(category_tensor, line_tensor):
    hidden = rnn.initHidden(1)
#     print(hidden.shape)

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
#         print(hidden.shape)
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories, 'lstm')

start = time.time()

# Keep track of losses for plotting
n_iters = 100000
print_every = 5000
plot_every = 1000 

current_lstm_loss = 0
all_lstm_losses = []
all_lstm_test_losses = []

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_lstm_loss += loss

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = 'âœ“' if guess == category else 'âœ— (%s)' % category
        print(
            '%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_lstm_losses.append(current_lstm_loss / plot_every)
        current_lstm_loss = 0

    # Compute loss based on test data
    if iter % plot_every == 0:
        test_loss = 0
        n_test_instances = 0
        for category in all_categories:
            category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
            n_test_instances = n_test_instances + len(test_data[category])
            for line in test_data[category]:
                line_tensor = Variable(lineToTensor(line))
                output = evaluate(line_tensor)
                test_loss = test_loss + criterion(output, category_tensor)
        all_lstm_test_losses.append(test_loss.item() / n_test_instances)

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_lstm_test_losses)

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories, 'gru')

start = time.time()

# Keep track of losses for plotting
n_iters = 100000
print_every = 5000
plot_every = 1000 

current_gru_loss = 0
all_gru_losses = []
all_gru_test_losses = []

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_gru_loss += loss

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = 'âœ“' if guess == category else 'âœ— (%s)' % category
        print(
            '%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_gru_losses.append(current_gru_loss / plot_every)
        current_gru_loss = 0

    # Compute loss based on test data
    if iter % plot_every == 0:
        test_loss = 0
        n_test_instances = 0
        for category in all_categories:
            category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
            n_test_instances = n_test_instances + len(test_data[category])
            for line in test_data[category]:
                line_tensor = Variable(lineToTensor(line))
                output = evaluate(line_tensor)
                test_loss = test_loss + criterion(output, category_tensor)
        all_gru_test_losses.append(test_loss.item() / n_test_instances)

plt.figure()
plt.plot(all_gru_test_losses)

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)

start = time.time()

# Keep track of losses for plotting
n_iters = 100000
print_every = 5000
plot_every = 1000 

current_loss = 0
all_losses = []
all_test_losses = []

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = 'âœ“' if guess == category else 'âœ— (%s)' % category
        print(
            '%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

    # Compute loss based on test data
    if iter % plot_every == 0:
        test_loss = 0
        n_test_instances = 0
        for category in all_categories:
            category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
            n_test_instances = n_test_instances + len(test_data[category])
            for line in test_data[category]:
                line_tensor = Variable(lineToTensor(line))
                output = evaluate(line_tensor)
                test_loss = test_loss + criterion(output, category_tensor)
        all_test_losses.append(test_loss.item() / n_test_instances)

plt.figure()
plt.plot(all_test_losses)

plt.plot(all_test_losses,'b',label='Linear Hidden Units')
plt.plot(all_lstm_test_losses,'g',label='LSTM Hidden Units')
plt.plot(all_gru_test_losses,'r',label='GRU Hidden Units')

plt.title('NLL vs Iterations')
plt.xlabel("Iterations per thousands")
plt.ylabel("NLL")
plt.legend()
plt.savefig("hidden_units_graph.png")
plt.figure()

