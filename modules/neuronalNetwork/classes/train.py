import json
from utils import tokenizeAndLower, bag_of_words
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from model import NeuronalNet

with open('../../../data/intents.json', 'r') as f:
    data = json.load(f)

# Caracteres especiales a ignorar
ignore_words = ['?', '.', '!']


tags = []
allwords = []
xy = []

for data_train in data:
    tags.append(data_train['tag'])

    for pattern in data_train['patterns']:
        # tokenize cada frase
        # separar la frase en cada palabra
        words_tokenized = tokenizeAndLower(pattern)

        # Guardar en todas las palabras las frases pertenecientes al tag
        allwords.extend(words_tokenized)

        # Guardar (tag, words_tokenized)
        # Para saber para cada tag que palabras le pertenecen
        xy.append((data_train['tag'], words_tokenized))


# remove ignored characters
# remove duplicates and sort
all_words = sorted(set(allwords))
tags = sorted(set(tags))

x_train = []
y_train = []

for (tag, tokenized_sentence) in xy:
    bag_words = bag_of_words(tokenized_sentence, all_words)

    x_train.append(bag_words)

    label = tags.index(tag)
    y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)

# parametros para entrenamiento
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(x_train[0])
hidden_size = 8
output_size = len(tags)


dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define model
model = NeuronalNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
