# flower-image-classifier

Link to download database: http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz

Link to download checkpoint for predict.py: https://www.dropbox.com/s/mk8lmff8qilsanr/checkpoint.pth?dl=0

Jupyter notebook: All cell can be run in sequence and the requirements are completed step by step.


Use train.py to train a model and save a checkpoint. Use predict.py to make prediction on the dataset images.


train.py: example python train.py flowers --arch vgg --lr 0.001 --units 2048 --epochs 5 --gpu
    parser.add_argument('data', type = str, help = 'Path to data') 
    parser.add_argument('--save_dir', '-S', type = str, default = '', help = 'checkpoint save directory') 
    parser.add_argument('--arch', '-A', type = str, default = 'vgg', help = 'CNN Model Architecture') 
    parser.add_argument('--lr','-L',  type = float, default = 0.001, help = 'Learning rate for the network') 
    parser.add_argument('--units', '-U', type = int, default = 2048, help = 'Number of hidden units ') 
    parser.add_argument('--epochs', '-E', type = int, default = 9, help = 'Epochs to be used') 
    parser.add_argument('--gpu', '-G', action='store_true')
    parser.add_argument('--print_graph', '-P', action='store_true')

predict.py: example python predict.py flowers/test/1/image_06743.jpg checkpoint.pth --gpu --top_k 5 --category_names cat_to_name.json
    parser.add_argument('image_path', type = str, help = 'Path to image') 
    parser.add_argument('checkpoint', type = str, help = 'checkpoint save directory') 
    parser.add_argument('--top_k', '-T', type = int, default = 1, help = 'Top K most likely classes') 
    parser.add_argument('--category_names','-C',  type = str, default ='', help = 'category names will allow user to use a mapping of categories to real names') 
    parser.add_argument('--gpu', '-G', action='store_true')
