import random
import json

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from modules.neuronalNetwork.classes.chatDataset import ChatDataset

# constants
from modules.neuronalNetwork.constants.constants import IGNORE_CHARACTERS, FILE_LOCATION

# numpy
import numpy as np

# util classes
from modules.neuronalNetwork.classes.model import NeuronalNet
from modules.neuronalNetwork.classes.utils import bag_of_words, tokenizeAndLower

# declarate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def getResponseFromMessage(inputMessage: str) -> str:
    # load intents
    with open('data/intents.json', 'r') as json_data:
        intents = json.load(json_data)

    # load model from file
    model_data = torch.load(FILE_LOCATION)

    # get params from model
    input_size = model_data["input_size"]
    hidden_size = model_data["hidden_size"]
    output_size = model_data["output_size"]
    all_words = model_data['all_words']
    tags = model_data['tags']
    model_state = model_data["model_state"]

    # define model
    model = NeuronalNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()

    # PASS MESSAGE THROUGHT MODEL
    # tokenize message
    tokenized_message = tokenizeAndLower(inputMessage)
    # bag of words
    bag_words = bag_of_words(tokenized_message, all_words)

    # newuronal networks inputs
    nn_inputs = torch.from_numpy(
        bag_words.reshape(1, bag_words.shape[0])).to(device)

    # GET OUTPUT MESSAGE
    # get nn output
    nn_output = model(nn_inputs)
    _, predicted = torch.max(nn_output, dim=1)

    # get response tag
    response_tag = tags[predicted.item()]

    # softmax the outputs
    probs = torch.softmax(nn_output, dim=1)
    prob = probs[0, predicted.item()]

    if prob.item() > 0.75:
        for intent in intents:
            if response_tag == intent["tag"]:
                return f"{random.choice(intent['responses'])}"
    else:
        return f"Disculpa no entiendo"


def trainModel():
    # load intents
    with open('data/intents.json', 'r') as json_data:
        model_training_data = json.load(json_data)

    tags = []
    allwords = []
    xy = []

    for data_train in model_training_data:
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
    all_words = [w for w in allwords if w not in IGNORE_CHARACTERS]

    # remove duplicates and sort
    all_words = sorted(set(allwords))
    tags = sorted(set(tags))

    # create training data
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

    # create training dataset
    dataset = ChatDataset(x_train, y_train)
    train_loader = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0)

    # Define model
    model = NeuronalNet(input_size, hidden_size, output_size).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train the model
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

    # save model
    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "all_words": all_words,
        "tags": tags
    }

    torch.save(data, FILE_LOCATION)
