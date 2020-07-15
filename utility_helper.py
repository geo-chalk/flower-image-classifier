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
import json
from collections import OrderedDict
from PIL import Image
from model_helper import validation



def load_data(data_dir):

    """ 
    Defines the Dataloaders based on the input data used. Firstly, the corresponding
    folders for train, validation and test data is located, then the transforms are defined, 
    then loaded and finally the datasets are defined.
    Input: data_dir
    Output: train_data -> dataset (to later get the class to index mapping), 
            trainloader, validationloader, testloader -> to use for training the model
    """

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    #Define transforms for the training, validation, and testing sets
    training_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    validation_testing_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=training_transform)
    valid_data = datasets.ImageFolder(valid_dir, transform=validation_testing_transform)
    test_data = datasets.ImageFolder(test_dir , transform=validation_testing_transform)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validationloader =  torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32) 

    return train_data, trainloader, validationloader, testloader


def print_input_arguments(arg):
    """
    Print all the arguments based on user inputs.
    Input: an argparse.ArgumentParser() object
    output: None
    """
    print('Inputs: ')
    print(f'Data Path: {arg.data}')
    print(f'Checkpoint save directory: {arg.save_dir}')
    print(f'CNN Model Architecture: {arg.arch}')  
    print(f'Learning rate for the network: {arg.lr}')
    print(f'Number of hidden units: {arg.units}')
    print(f'Epochs to be used: {arg.epochs}')    
    print(f'GPU device selected: {arg.gpu}')  
    print(f'Print graph: {arg.print_graph}', end='\n'*3)  

def print_model(model):
    """
    Print the model's hidden layers along with the classifier created by the user.
    Input: A torchvision model
    output: None
    """    
    print('Model Used: ')
    print(model, end='\n'*5)
    
    
def model_validation(device, model, testloader, criterion, train_loss_toprint, valid_loss_toprint, valid_accuracy_toprint):
    """
    Perform a validation on the train set and print a graph showing the progress of the training loss, validation loss and validation accuracy.
    Inputs: device, model, testloader, criterion, train_loss_toprint, valid_loss_toprint, test_accuracy_toprint
    output: None (prints losses and graphs)
    """
    # Do validation on the test set, print the validation and train loss along with the validation accuracy
    model.to(device)
    with torch.no_grad():
        test_loss, accuracy = validation(model, testloader, criterion, device)  

    print("Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
          "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
    

#     fig, ax1 = plt.subplots(figsize=(12,7))
#     x = np.arange(len(train_loss_toprint))
#     len(train_loss_toprint)
#     ax1.plot(x, train_loss_toprint, label="Train Loss")
#     ax1.plot(x, valid_loss_toprint, label="Validation Loss")
#     ax1.set_ylim([float(min(train_loss_toprint))*0.85,float(max(train_loss_toprint))*1.2])
#     plt.legend(loc=2)
#     ax2 = ax1.twinx()
#     ax2.plot(x, valid_accuracy_toprint, label="Validation Accuracy", color='red')
#     ax2.set_ylim([float(min(valid_accuracy_toprint))*0.85,float(max(valid_accuracy_toprint))*1.2])
#     plt.legend(loc=1)

#     #fig.tight_layout()
    
#     fig.savefig('Graph.png')
#     plt.close(fig)


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    im = Image.open(image)
    im = im.resize((256,256))
    value = 0.5*(256-224)
    im = im.crop((value,value,256-value,256-value))
    im = np.array(im)/255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    im = (im - mean) / std

    return im.transpose(2,0,1) 

def get_key(val, model): 
    for key, value in model.class_to_idx.items(): 
         if val == value: 
             return key 
    return "key doesn't exist"

def predict(image_path, model, device, top_k=1, cat_to_name=''):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.to(device)
    
    inp = process_image(image_path)
    # tranfer to tensor
    inp = torch.from_numpy(np.array([inp])).float()
    inp = inp.to(device)
    model.eval()
    with torch.no_grad():
        output = model.forward(inp)    

    # TODO: Calculate the class probabilities (softmax) for img
    ps = torch.exp(output).data 
    ps = ps.topk(top_k)

    
    probabilities = [float(i) for i in ps[0][0]]
    labels = [get_key(int(i),model) for i in ps[1][0]]
   

    if cat_to_name:
        with open('cat_to_name.json', 'r') as f:
            cat_to_name = json.load(f)
        labels=[cat_to_name[i] for i in labels] 


    # Plot the image and probabilities
    fig, (ax1, ax2) = plt.subplots(figsize=(15,6), ncols=2)
    
    #image = PIL.Image.open(image)
    image = Image.open(image_path)
    ax1.imshow(image)
    label = image_path.split('/')[2]
    ax1.set_title(f'Flower: {cat_to_name[str(label)]}')

    ax2.barh(np.arange(top_k), probabilities)
    ax2.set_yticks(np.arange(top_k))
    ax2.set_yticklabels(labels)
    ax2.set_xlim((0, 1.1))
    xlabels = [f'{v}' for v in range(0,110,20)]
    plt.xticks(np.arange(0,1.1,0.2),xlabels)
    ax2.invert_yaxis()  # labels read top-to-bottom
    ax2.set_xlabel('Prediction')
    ax2.set_title(f'Best Prediction: {labels[0]}')
    fig.tight_layout(pad=3.0)  

    plt.show()        
    return probabilities, labels

 