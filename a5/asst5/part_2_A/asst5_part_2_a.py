# -*- coding: utf-8 -*-

import json
import numpy as np

# load metadata
with open("train.json", 'r') as f:
    metadata = json.load(f)
n_claims = len(metadata)

import torch
import torch.nn as nn
import numpy as np
import random


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        return hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


class Classifier(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Classifier, self).__init__()
        self.hidden_size = hidden_size
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, hidden):
        hidden = self.h2o(hidden)
        output = self.softmax(hidden)
        return output

def preprocess_articles():
    from nltk.tokenize import sent_tokenize
    import nltk
    nltk.download('punkt')
    from sklearn.feature_extraction.text import TfidfVectorizer

    # load metadata
    with open("train.json", 'r') as f:
        metadata = json.load(f)
    n_claims = len(metadata)

    # load related articles for each claim
    relevant_sentences = []
    for id in range(n_claims):

        if id % 500 == 0:
            print("Claims preprocessed: ", id)

        # retrieve related articles
        related_articles = metadata[id]['related_articles']
        articles = ""
        for article_id in related_articles:
            filename = "train_articles/" + str(article_id) + ".txt"
            # concatenate related articles
            with open(filename, 'r') as text_file:
                text = text_file.read()
                articles = articles + "\n" + text

        # split articles into sentences
        sentences = sent_tokenize(articles)

        # append claim to articles
        sentences.append(metadata[id]['claim'])

        # vectorize sentences based on tf-idf
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(sentences)

        # measure similarity between claim and each sentence
        similarity = X[-1, :] @ np.transpose(X[:-2, :])
        similarity = similarity.todense()

        # find top 5 sentences with greatest similarity
        sorted_index = np.argsort(similarity)
        top_sentences = []
        for i in range(1, min(5, sorted_index.shape[1]) + 1):
            top_sentences.append(sentences[sorted_index[0, -i]])
        relevant_sentences.append(top_sentences)

    return metadata, relevant_sentences


metadata, relevant_sentences = preprocess_articles()
print("Metadata of claim 0:")
print(metadata[0]['claim'])
print("Relevant sentences of claim 0:")
print(relevant_sentences[0])

from bpemb import BPEmb

n_embedding_dims = 50
bpemb_en = BPEmb(lang="en", dim=n_embedding_dims)


def sampleClaim(metadata, model):
    id = random.randint(0, len(metadata) - 1)
    claim = metadata[id]["claim"]
    
    if model == 'ii':
      claimant = metadata[id]["claimant"]
      claim += ' '+ claimant
      
    if model == 'iii':
      claimant = metadata[id]["claimant"]
      claim += ' '+ claimant
      for sentence in relevant_sentences[id]:
        claim += ' '+ sentence
        
    embedding = bpemb_en.embed(claim)
    embedding = np.reshape(embedding, (embedding.shape[0], 1, embedding.shape[1]))
    label = metadata[id]["label"]
    label_tensor = torch.tensor([label], dtype=torch.long)
    claim_tensor = torch.tensor(embedding, dtype=torch.float)
    return claim_tensor, label_tensor, claim, label, id

# !pip install bpemb

# ! unzip '/content/drive/My Drive/train.zip'

def train(category_tensor, line_tensor, update=True):
    rnnOptimizer.zero_grad()
    classifierOptimizer.zero_grad()

    hidden = rnn.initHidden()
    for i in range(line_tensor.size()[0]):
        hidden = rnn(line_tensor[i], hidden)
    output = classifier(hidden)

    loss = criterion(output, category_tensor)
    if update:
        loss.backward()
        rnnOptimizer.step()
        classifierOptimizer.step()

    return output, loss.item()

import time
import math

n_hidden = 128
n_categories = 3
rnn = RNN(n_embedding_dims, n_hidden)
classifier = Classifier(n_hidden, n_categories)

criterion = nn.NLLLoss()
learning_rate = 1e-4
rnnOptimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
classifierOptimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)

n_iters = 100000
plot_every = 1000

# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def iterTrain(model):
  start = time.time()

  train_data = metadata[:10000]
  test_data = metadata[10000:]
  train_cumulative_loss = 0
  test_cumulative_loss = 0
  average_train_accuracy = 0
  average_test_accuracy = 0
  all_train_losses = []
  all_train_accuracies = []
  all_test_losses = []
  all_test_accuracies = []
  count = 0
  
  for iter in range(1, n_iters + 1):
      train_line_tensor, train_category_tensor, train_line, train_category, train_id = sampleClaim(train_data, model)
      train_output, train_loss = train(train_category_tensor, train_line_tensor)
      top_train_value, top_train_index = train_output.topk(1)
      train_guess_category = top_train_index[0].item()
      train_cumulative_loss += train_loss
      train_accuracy = 1 if train_guess_category == train_category else 0
      average_train_accuracy = (average_train_accuracy * count + train_accuracy) / (count + 1)

      test_line_tensor, test_category_tensor, test_line, test_category, test_id = sampleClaim(test_data, model)
      test_output, test_loss = train(test_category_tensor, test_line_tensor, update=False)
      top_test_value, top_test_index = test_output.topk(1)
      test_guess_category = top_test_index[0].item()
      test_cumulative_loss += test_loss
      test_accuracy = 1 if test_guess_category == test_category else 0
      average_test_accuracy = (average_test_accuracy * count + test_accuracy) / (count + 1)
      count += 1

      # Add current loss avg to list of losses
      if iter % plot_every == 0:
          train_correct = 'âœ“' if train_guess_category == train_category else 'âœ— (%s)' % train_category
          print('Train: %d  %d%% (%s) average_accuracy=%.4f average_loss=%.4f %s / %s %s' % (
          iter, iter / n_iters * 100, timeSince(start), average_train_accuracy, train_cumulative_loss / plot_every,
          train_line, train_guess_category, train_correct))
          test_correct = 'âœ“' if test_guess_category == test_category else 'âœ— (%s)' % test_category
          print('Test: %d  %d%% (%s) average_accuracy=%.4f average_loss=%.4f %s / %s %s' % (
          iter, iter / n_iters * 100, timeSince(start), average_test_accuracy, test_cumulative_loss / plot_every,
          test_line, test_guess_category, test_correct))
          all_train_losses.append(train_cumulative_loss / plot_every)
          all_train_accuracies.append(average_train_accuracy)
          all_test_losses.append(test_cumulative_loss / plot_every)
          all_test_accuracies.append(average_test_accuracy)
          train_cumulative_loss = 0
          average_train_accuracy = 0
          test_cumulative_loss = 0
          average_test_accuracy = 0
          count = 0
  return all_train_accuracies, all_test_accuracies

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

all_i_train_accuracies, all_i_test_accuracies = iterTrain('i')
plt.figure()
plt.plot(all_i_train_accuracies)
plt.plot(all_i_test_accuracies)

all_ii_train_accuracies, all_ii_test_accuracies = iterTrain('ii')
plt.figure()
plt.plot(all_ii_train_accuracies)
plt.plot(all_ii_test_accuracies)

all_iii_train_accuracies, all_iii_test_accuracies = iterTrain('iii')
plt.figure()
plt.plot(all_iii_train_accuracies)
plt.plot(all_iii_test_accuracies)

plt.plot(all_i_train_accuracies,'b',label='I')
plt.plot(all_ii_train_accuracies,'g',label='II')
plt.plot(all_iii_train_accuracies,'r',label='III')


plt.title('Accuracy vs Iterations')
plt.xlabel("Iterations")
plt.ylabel("Training Accuracy")
plt.legend()
plt.savefig("part_2_A_training_graph.png")
plt.figure()

plt.plot(all_i_test_accuracies,'b',label='I')
plt.plot(all_ii_test_accuracies,'g',label='II')
plt.plot(all_iii_test_accuracies,'r',label='III')


plt.title('Accuracy vs Iterations')
plt.xlabel("Iterations")
plt.ylabel("Test Accuracy")
plt.legend()
plt.savefig("part_2_A_test_graph.png")
plt.figure()

