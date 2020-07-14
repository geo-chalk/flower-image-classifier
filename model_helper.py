# Imports here
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import time



def create_model(arch):
    """
    Creates the network model based on user input. 
    Input: input arguments ('vgg', 'alexnet')
    Output: The torchvision pre-trained model
    """
    if arch == 'vgg':
        arch = 'vgg16'

    model = getattr(models, str(arch))(pretrained=True)
    return model


def create_classifier(model, hidden):
    """
    Return the feed-forward network as a classifier using ReLU activations and dropout.
    The input size is defined based on the first ReLU activation encounter of the pre-trained model.
    Two hidden layers are used, one with the actual number of hidden units and another one with the number divided by two.
    Input: the model and the number of the hidden Layers
    Output: a new, untrained feed-forward network as a classifier
    """
    # Find the model's input features:
    for item in model.classifier:
        if type(item) is torch.nn.modules.linear.Linear:
            input_size = item.in_features
            break
    # Based on this input create the 
    classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(input_size, hidden)),
                            ('dropout1', nn.Dropout(p=0.5)),
                            ('relu1', nn.ReLU()),
                            ('fc2', nn.Linear(hidden, int(hidden/2))),
                            ('relu2', nn.ReLU()),
                            ('dropout2', nn.Dropout(p=0.5)),
                            ('fc3', nn.Linear(int(hidden/2), 102)),    
                            ('output', nn.LogSoftmax(dim=1))
                            ]))  
    return classifier,  input_size                          


# Implement a function for the validation pass
def validation(model, testloader, criterion, device):
    """
    Function to return the validation loss and the accuracy based on the validation dataset
    Input: current Model, validation, criterion, current device
    Output: current test loss and accuracy of the validation dataset 
    """
    test_loss = 0
    accuracy = 0
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy 

def train_model(model, device, epochs, trainloader, lr, validationloader):
    """
    Trains the model based on the inputs given.
    """
    # Initialize Values to display
    print_every = 16
    steps = 0
    train_loss_toprint = []
    valid_loss_toprint = []
    valid_accuracy_toprint = []
    total_epochs = 1
    
    # Set the criterion
    criterion = nn.NLLLoss()
    running_loss = 0 

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr)    
    # Load the model to the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    start = time.time()
    for e in range(epochs):
        epoch_time = time.time()
        if total_epochs:
            print(f'Epoch: {total_epochs}')
        else:
            print('Training has started')

        total_epochs += 1
        #Dropout enabled
        model.train()
        for inputs, labels in trainloader:
            steps += 1

            # Move input and label tensors to the GPU
            inputs, labels = inputs.to(device), labels.to(device)


            # Set optimizer grads to zero so we dont add them add in each loop
            optimizer.zero_grad()

            # Do a forward pass, backpropagation and a grad step
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Calculate Current loss
            running_loss += loss.item()


            if steps % print_every == 0:
                #turn off dropout for testing validation
                model.eval()

                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    test_loss, accuracy = validation(model, validationloader, criterion, device)            

                train_loss_toprint.append(running_loss/print_every)
                valid_loss_toprint.append(test_loss/len(validationloader))
                valid_accuracy_toprint.append(accuracy/len(validationloader))
                print("Step: {:5} Epoch: {}/{}... ".format(steps,e+1, epochs),
                    "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(test_loss/len(validationloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validationloader)))

                running_loss = 0 

                # Turn Dropout back on
                model.train()

        print(f"Epoch Time Elapsed: {(time.time() - epoch_time)/60}")  
        print(f"Total Time Elapsed: {(time.time() - start)/60}\n")  


    print(f'All done!!! \nTotal Time: {(time.time() - start)/60}')
    
    return model, criterion, optimizer, train_loss_toprint, valid_loss_toprint, valid_accuracy_toprint 


def load_checkpoint(filepath):
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
        
    
        
    checkpoint = torch.load(filepath, map_location=map_location)

    if checkpoint['arch'] == 'vgg':
        checkpoint['arch'] = 'vgg16'
    model = getattr(models, str(checkpoint['arch']))(pretrained=True)
    #checkpoint = torch.load(filepath)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = optim.Adam(model.classifier.parameters(), checkpoint['learning_rate'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    learning_rate = checkpoint['learning_rate']
    train_loss_toprint = checkpoint['train_loss_toprint']
    valid_loss_toprint = checkpoint['test_loss_toprint']
    valid_accuracy_toprint = checkpoint['test_accuracy_toprint']
        
    return model, optimizer   


