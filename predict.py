import torchvision
import ast
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
from torchvision import datasets
from torch import __version__
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim
import argparse
from torch.autograd import Variable
import json


def load_checkpoint(filepath='checkpoint.pth'):
    ''' Load the checkpoint and build the model
        returns the model
    '''

    models_list = {
                   'vgg13': models.vgg13(pretrained=True),
                   'vgg16': models.vgg16(pretrained=True)
    }
    checkpoint = torch.load(filepath)
    model = models_list[checkpoint['arch']]
    for param in model.parameters():
        param.requires_grad = False
    model.class_to_idx = checkpoint['classes to indices mapping']
    model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(checkpoint['input_size'], checkpoint['hidden_size'])),
                          ('relu', nn.ReLU()),
                          ('dropout',nn.Dropout(0.2)),
                          ('fc2', nn.Linear(checkpoint['hidden_size'], checkpoint['output_size'])),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model.load_state_dict(checkpoint['state_dict'])

    return model


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Load image using Image
    image = Image.open(image)
    
    # Resize, crop and normalize the loaded image and transform into a tensor
    transformation = transforms.Compose([
                         transforms.Resize(256),
                         transforms.CenterCrop(224),
                         transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    processed_image = transformation(image)
    
    return processed_image


def predict(image_path, model, device='cuda', category_names='cat_to_name.json', topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    model.eval()
    model.to(device)
    img = process_image(image_path)
    # Calculate the class probabilities (softmax) for img
    with torch.no_grad():
        output = model.forward(img.unsqueeze(0).cuda())

    ps = torch.exp(output)
    top_p, top_class = ps.topk(topk, dim=1)
    probs = probs.cpu().numpy().tolist()[0]
    classes = classes.cpu().numpy().tolist()[0]
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    classes = [idx_to_class[x] for x in classes]
    flowers = [cat_to_name[str(x)] for x in classes]
    return probs, flowers


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Predict flower name from image of a flower')
    parser.add_argument('image_path', help='image  path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--top_k', help='K most likely classes')
    parser.add_argument('--category_names',
                        help='mapping of classes to flower names')
    parser.add_argument('--gpu', help='use gpu if available')
    args = parser.parse_args()
    topk = args.top_k if args.top_k else 5
    category_names = args.category_names if args.category_names else '/home/workspace/aipnd-project/cat_to_name.json'
    device = 'cuda' if args.gpu else 'cpu'

    model = load_checkpoint(filepath=args.checkpoint)
    probs, flowers = predict(args.image_path, model, topk=topk,
                             category_names=category_names, device=device) 