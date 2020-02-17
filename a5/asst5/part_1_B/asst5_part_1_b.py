# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import math
import random

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1  # Plus EOS marker


def findFiles(path): return glob.glob(path)


# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


# Build the category_lines dictionary, a list of lines per category
category_lines = {}
train_data = {}
test_data = {}
all_categories = []
for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    random.shuffle(lines)
    train_data[category] = lines[0:int(math.floor(0.8 * len(lines)))]
    test_data[category] = lines[int(math.floor(0.8 * len(lines))) + 1:]
    category_lines[category] = lines

n_categories = len(all_categories)

if n_categories == 0:
    raise RuntimeError('Data not found. Make sure that you downloaded data '
                       'from https://download.pytorch.org/tutorial/data.zip and extract it to '
                       'the current directory.')

print('# categories:', n_categories, all_categories)
print(unicodeToAscii("O'NÃ©Ã l"))

import torch
import torch.nn as nn
from torch.autograd import Variable


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, model = 'i'):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.model = model
        
        if self.model == 'ii':
          self.i2h_0 = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
          self.i2o_0 = nn.Linear(n_categories + input_size + hidden_size, output_size)
          self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
          self.i2o = nn.Linear(input_size + hidden_size, output_size)
          
        elif self.model == 'iii':
          self.i2h = nn.Linear(n_categories + hidden_size, hidden_size)
          self.i2o = nn.Linear(n_categories + hidden_size, output_size)
          
        elif self.model == 'iv':
          self.i2h_0 = nn.Linear(n_categories + hidden_size, hidden_size)
          self.i2o_0 = nn.Linear(n_categories + hidden_size, output_size)
          self.i2h = nn.Linear(hidden_size, hidden_size)
          self.i2o = nn.Linear(hidden_size, output_size) 
          
        else:
          self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
          self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)    
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden, time_step):
#       print(time_step)
      if self.model == 'ii':
          if time_step == 0:
            input_combined = torch.cat((category, input, hidden), 1)  
#             print(input_combined.shape, hidden.shape, input.shape, category.shape)
            hidden = self.i2h_0(input_combined)
#             print(hidden.shape)
            output = self.i2o_0(input_combined)
          else:
            input_combined = torch.cat((input, hidden), 1) 
            hidden = self.i2h(input_combined)
            output = self.i2o(input_combined)              
        
      elif self.model == 'iii':
        input_combined = torch.cat((category, hidden), 1)  
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)

      elif self.model == 'iv':
        if time_step == 0:
          input_combined = torch.cat((category, hidden), 1)
          hidden = self.i2h_0(input_combined)
          output = self.i2o_0(input_combined)
        else: 
          hidden = self.i2h(hidden)
          output = self.i2o(hidden)

      else:  
        input_combined = torch.cat((category, input, hidden), 1)   
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)

      output_combined = torch.cat((hidden, output), 1)
#       print(output_combined.shape)
      output = self.o2o(output_combined)
      output = self.dropout(output)
      output = self.softmax(output)
      return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

import random


# Random item from a list
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


# Get a random category and random line from that category
def randomTrainingPair():
    category = randomChoice(all_categories)
    line = randomChoice(train_data[category])
    return category, line

# One-hot vector for category
def categoryTensor(category):
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return tensor


# One-hot matrix of first to last letters (not including EOS) for input
def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor


# LongTensor of second letter to end (EOS) for target
def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1)  # EOS
    return torch.LongTensor(letter_indexes)


# Make category, input, and target tensors from a random category, line pair
def randomTrainingExample():
    category, line = randomTrainingPair()
    category_tensor = categoryTensor(category)
    input_line_tensor = inputTensor(line)
    target_line_tensor = targetTensor(line)
    return category_tensor, input_line_tensor, target_line_tensor

def train(category_tensor, input_line_tensor, target_line_tensor):
    target_line_tensor.unsqueeze_(-1)
    hidden = rnn.initHidden()

    rnn.zero_grad()

    loss = 0

    for i in range(input_line_tensor.size(0)):
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden, i)
        l = criterion(output, target_line_tensor[i])
        loss += l

    loss.backward()

    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item() / input_line_tensor.size(0)

import time
import math


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# Just return an output given a line
def evaluate(category_tensor, input_line_tensor, target_line_tensor):
    target_line_tensor.unsqueeze_(-1)
    hidden = rnn.initHidden()
    loss = 0
    for i in range(input_line_tensor.size()[0]):
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden, i)
        loss += criterion(output, target_line_tensor[i])
    return output, loss.item() / input_line_tensor.size(0)

# n_letters

criterion = nn.NLLLoss()

learning_rate = 0.0005

n_iters = 100000
print_every = 5000
plot_every = 1000
all_losses = []
all_test_losses = []
total_loss = 0  # Reset every plot_every iters

rnn = RNN(n_letters, 128, n_letters, model = 'i')

start = time.time()

for iter in range(1, n_iters + 1):
    output, loss = train(*randomTrainingExample())
    total_loss += loss

    if iter % print_every == 0:
        print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))

    if iter % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        total_loss = 0

    # Compute loss based on test data
    if iter % plot_every == 0:
        total_test_loss = 0
        n_test_instances = 0
        for category in all_categories:
            category_tensor = Variable(categoryTensor(category))
            n_test_instances = n_test_instances + len(test_data[category])
            for line in test_data[category]:
                input_line_tensor = Variable(inputTensor(line))
                target_line_tensor = Variable(targetTensor(line))
                output, test_loss = evaluate(category_tensor, input_line_tensor, target_line_tensor)
                total_test_loss += test_loss
        all_test_losses.append(total_test_loss / n_test_instances)

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_test_losses)

criterion = nn.NLLLoss()

learning_rate = 0.0005

n_iters = 100000
print_every = 5000
plot_every = 1000
all_ii_losses = []
all_ii_test_losses = []
total_ii_loss = 0  # Reset every plot_every iters

rnn = RNN(n_letters, 128, n_letters, model = 'ii')

start = time.time()

for iter in range(1, n_iters + 1):
    output, loss = train(*randomTrainingExample())
    total_ii_loss += loss

    if iter % print_every == 0:
        print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))

    if iter % plot_every == 0:
        all_ii_losses.append(total_ii_loss / plot_every)
        total_ii_loss = 0

    # Compute loss based on test data
    if iter % plot_every == 0:
        total_test_loss = 0
        n_test_instances = 0
        for category in all_categories:
            category_tensor = Variable(categoryTensor(category))
            n_test_instances = n_test_instances + len(test_data[category])
            for line in test_data[category]:
                input_line_tensor = Variable(inputTensor(line))
                target_line_tensor = Variable(targetTensor(line))
                output, test_loss = evaluate(category_tensor, input_line_tensor, target_line_tensor)
                total_test_loss += test_loss
        all_ii_test_losses.append(total_test_loss / n_test_instances)

plt.figure()
plt.plot(all_ii_test_losses)

criterion = nn.NLLLoss()

learning_rate = 0.0005

n_iters = 100000
print_every = 5000
plot_every = 1000
all_iii_losses = []
all_iii_test_losses = []
total_iii_loss = 0  # Reset every plot_every iters

rnn = RNN(n_letters, 128, n_letters, model = 'iii')

start = time.time()

for iter in range(1, n_iters + 1):
    output, loss = train(*randomTrainingExample())
    total_iii_loss += loss

    if iter % print_every == 0:
        print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))

    if iter % plot_every == 0:
        all_iii_losses.append(total_iii_loss / plot_every)
        total_iii_loss = 0

    # Compute loss based on test data
    if iter % plot_every == 0:
        total_test_loss = 0
        n_test_instances = 0
        for category in all_categories:
            category_tensor = Variable(categoryTensor(category))
            n_test_instances = n_test_instances + len(test_data[category])
            for line in test_data[category]:
                input_line_tensor = Variable(inputTensor(line))
                target_line_tensor = Variable(targetTensor(line))
                output, test_loss = evaluate(category_tensor, input_line_tensor, target_line_tensor)
                total_test_loss += test_loss
        all_iii_test_losses.append(total_test_loss / n_test_instances)

plt.figure()
plt.plot(all_iii_test_losses)

criterion = nn.NLLLoss()

learning_rate = 0.0005

n_iters = 100000
print_every = 5000
plot_every = 1000
all_iv_losses = []
all_iv_test_losses = []
total_iv_loss = 0  # Reset every plot_every iters

rnn = RNN(n_letters, 128, n_letters, model = 'iv')

start = time.time()

for iter in range(1, n_iters + 1):
    output, loss = train(*randomTrainingExample())
    total_iv_loss += loss

    if iter % print_every == 0:
        print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))

    if iter % plot_every == 0:
        all_iv_losses.append(total_iv_loss / plot_every)
        total_iv_loss = 0

    # Compute loss based on test data
    if iter % plot_every == 0:
        total_test_loss = 0
        n_test_instances = 0
        for category in all_categories:
            category_tensor = Variable(categoryTensor(category))
            n_test_instances = n_test_instances + len(test_data[category])
            for line in test_data[category]:
                input_line_tensor = Variable(inputTensor(line))
                target_line_tensor = Variable(targetTensor(line))
                output, test_loss = evaluate(category_tensor, input_line_tensor, target_line_tensor)
                total_test_loss += test_loss
        all_iv_test_losses.append(total_test_loss / n_test_instances)

plt.figure()
plt.plot(all_iv_test_losses)

plt.plot(all_test_losses,'b',label='I')
plt.plot(all_ii_test_losses,'g',label='II')
plt.plot(all_iii_test_losses,'r',label='III')
plt.plot(all_iv_test_losses,'y',label='IV')

plt.title('NLL vs Iterations')
plt.xlabel("Iterations per thousands")
plt.ylabel("NLL")
plt.legend()
plt.savefig("Feeding_inputs_graph.png")
plt.figure()

