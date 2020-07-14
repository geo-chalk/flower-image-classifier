# PROGRAMMER: George Chalkiopoulos
# DATE CREATED: 06/07/2020                                   
# REVISED DATE: 
# PURPOSE: Creates a function that retrieves the following command line inputs 
#          from the user using the Argparse Python module. If the user fails to 
#          provide some or all of the 3 inputs, then the default values are
#          used for the missing inputs. Command Line Arguments:
#     1. CNN Model Architecture as --arch with default value 'vgg'. Options: vgg, densenet121
#     2. Learning rate to be used in order to train the network. default value would be "0.001"
#     3. Number of hidden units in a form of a list. default values are: {vgg: [2048, 1024], densenet121: }
#     4. Number of training epochs. Default is 4
#
##
# Imports python modules
import argparse


def get_input_args():
    """
    Creates a function that retrieves the following command line inputs 
    from the user using the Argparse Python module. If the user fails to 
    provide some or all of the inputs, then the default values are
    used for the missing inputs. Command Line Arguments:
        1. Path to Data, mandatory argument. Default option is /flowers
        2. Checkpoint save directory. Default option is current folder.
        3. CNN Model Architecture as --arch with default value 'vgg'. Options: vgg, alexnet
        4. Learning rate to be used in order to train the network. default value would be "0.001"
        5. Number of hidden units. default values are: {vgg: 2048 , densenet121: }
        6. Number of training epochs. Default is 4.
        7. GPU is a boolean with will return True if called
        8. print_graph will print a graph of the training and validation process.
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    # Create command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('data', type = str, help = 'Path to data') 
    parser.add_argument('--save_dir', '-S', type = str, default = '', help = 'checkpoint save directory') 
    parser.add_argument('--arch', '-A', type = str, default = 'vgg', help = 'CNN Model Architecture: vgg or alexnet') 
    parser.add_argument('--lr','-L',  type = float, default = 0.001, help = 'Learning rate for the network') 
    parser.add_argument('--units', '-U', type = int, default = 2048, help = 'Number of hidden units ') 
    parser.add_argument('--epochs', '-E', type = int, default = 3, help = 'Epochs to be used') 
    parser.add_argument('--gpu', '-G', action='store_true')
    parser.add_argument('--print_graph', '-P', action='store_true')
    
    # Replace None with parser.parse_args() parsed argument collection that 
    # you created with this function 
    return parser.parse_args()


def get_predict_input_args():
    """
    Creates a function that retrieves the following command line inputs 
    from the user using the Argparse Python module. If the user fails to 
    provide some or all of the inputs, then the default values are
    used for the missing inputs. Command Line Arguments:
        1. Path to image, mandatory argument. 
        2. Checkpoint load directory. Default option is current folder.
        3. Top K most likely classes. default is 1
        4. category names will allow user to use a mapping of categories to real names
        5. GPU is a boolean with will return True if called

    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    # Create command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('image_path', type = str, help = 'Path to image') 
    parser.add_argument('checkpoint', type = str, help = 'checkpoint save directory') 
    parser.add_argument('--top_k', '-T', type = int, default = 1, help = 'Top K most likely classes') 
    parser.add_argument('--category_names','-C',  type = str, default ='', help = 'category names will allow user to use a mapping of categories to real names') 
    parser.add_argument('--gpu', '-G', action='store_true')

    
    # Replace None with parser.parse_args() parsed argument collection that 
    # you created with this function 
    return parser.parse_args()
