import random
import json

import torch

# util classes
from modules.neuronalNetwork.classes.model import NeuronalNet
from modules.neuronalNetwork.classes.utils import bag_of_words, tokenizeAndLower

# declarate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def getResponseFromMessage(inputMessage: str) -> str:
    # file name
    FILE_NAME = 'modules/neuronalNetwork/nnModel/data.pth'

    # load intents
    with open('data/intents.json', 'r') as json_data:
        intents = json.load(json_data)

    # load model from file
    model_data = torch.load(FILE_NAME)

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
        return f"I do not understand..."
