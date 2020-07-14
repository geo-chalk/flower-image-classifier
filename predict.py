# PROGRAMMER: George Chalkiopoulos
# DATE CREATED: 06/07/2020                                 
# REVISED DATE: 
# PURPOSE: predicts a flower name fron an image using a trained network.
#
# Use argparse Expected Call with <> indicating expected user input:
#      predict.py flowers/test/1/image_06743.jpg checkpoint.pth --gpu --top_k <number if top k classes> --category_names <cat to name json file>
#   Example call:
#    predict.py flowers/test/1/image_06743.jpg checkpoint.pth --gpu --top_k 5 --category_names cat_to_name.json
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

from input_args import get_predict_input_args
from model_helper import load_checkpoint
from utility_helper import process_image, predict

def main():
    # get the input arguments
    in_arg = get_predict_input_args()
    #print_input_arguments(in_arg)
   
    # Load the model we saved during training
    model, optimizer = load_checkpoint(in_arg.checkpoint)
    print(model, optimizer, end='\n'*5)
    
    process_image(in_arg.image_path)
    if in_arg.gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = 'cpu'
    print(f"Device used: {device}", end='\n'*5)  
    
    probabilities, labels =  predict(in_arg.image_path, model, device, top_k=in_arg.top_k, cat_to_name=in_arg.category_names) 
    
    print(f'Prediction Results for top_k = {in_arg.top_k}:\n')
    identifier = 'name' if in_arg.category_names else 'label'
    for probability, flower in zip(probabilities,labels):            
        print('Predicted flower {}: {:20} Probability: {}'.format(identifier,flower,probability))
    
# Call to main function to run the program
if __name__ == "__main__":
    main()
    
