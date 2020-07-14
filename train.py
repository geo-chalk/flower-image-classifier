# PROGRAMMER: George Chalkiopoulos
# DATE CREATED: 06/07/2020                                 
# REVISED DATE: 
# PURPOSE: Trains a network on a dataset of flower images.
#
# Use argparse Expected Call with <> indicating expected user input:
#      python train.py data_directory --arch <architecture> --lr <learning rate> --units <number of hidden units> --epochs <number of epochs>
#      or
#      python train.py data_directory -A <architecture> -L <learning rate> -U <number of hidden units> -E <number of epochs>
#   Example call:
#    python train.py flowers --arch vgg --lr 0.003 --units [4096,1024] --epochs 9
##

import matplotlib.pyplot as plt
import numpy as np
import time
import json
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import os


# Imports functions created for this program
from input_args import get_input_args
from utility_helper import load_data, print_input_arguments, print_model, model_validation
from model_helper import create_model, create_classifier, train_model


def main():

    # get the input arguments
    in_arg = get_input_args()
    print_input_arguments(in_arg)

    # load the train dataset (to get the class to idx mapping) and the dataloaders for training validation and test
    train_data, trainloader, validationloader, testloader = load_data(in_arg.data)
    
    # Load the categry label to category name mapping
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    # Load the model
    model = create_model(in_arg.arch)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # Get the updated classifier based on the hidden layers parsed
    classifier, input_size = create_classifier(model, in_arg.units)
    model.classifier = classifier
    print_model(model)
    
    # Chose the model based on the "GPU" input
    if in_arg.gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = 'cpu'
    print(f"Device used: {device}")

    # Start training the model
    model, criterion, optimizer, train_loss_toprint, valid_loss_toprint, valid_accuracy_toprint  = train_model(model, device, in_arg.epochs, trainloader, in_arg.lr, validationloader)
    
    if in_arg.print_graph:
        model_validation(device, model, testloader, criterion, train_loss_toprint, valid_loss_toprint, valid_accuracy_toprint)
    
    checkpoint = {'input_size': input_size,
              'output_size': 102,
              'arch': in_arg.arch,
              'learning_rate': in_arg.lr,
              'batch_size': 64,
              'classifier' : classifier,
              'epochs': in_arg.epochs,
              'optimizer': optimizer.state_dict(),
              'state_dict': model.state_dict(),
              'class_to_idx': train_data.class_to_idx,
              'train_loss_toprint' :train_loss_toprint,
              'test_loss_toprint' : valid_loss_toprint,
              'test_accuracy_toprint' : valid_accuracy_toprint,
              'criterion' : criterion}


    if in_arg.save_dir:
        os.mkdir(in_arg.save_dir)
        torch.save(checkpoint, in_arg.save_dir + '/checkpoint.pth')
    else:
        torch.save(checkpoint, 'checkpoint.pth')


# Call to main function to run the program
if __name__ == "__main__":
    main()
