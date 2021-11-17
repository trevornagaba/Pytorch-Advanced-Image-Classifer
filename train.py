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
from torch.autograd import Variable
import argparse


def load_data(data_dir):
    '''Load data'''

    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])]) 

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])


    # TODO: Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

    return train_dataset, valid_dataset, test_dataset


def build_model(arch='vgg16', hidden_units=4096, lr=0.003):
    models_list = {
                   'vgg13': models.vgg13(pretrained=True),
                   'vgg16': models.vgg16(pretrained=True)}

    model = models_list[arch]

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(p=0.2)),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

    return model, criterion, optimizer

def train_model(model, trainset, validset, validation, epochs=5,
                device='cuda'):
    
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=True)
    running_loss = 0
    steps=0
    print_every=5
    train_losses, test_losses = [], []

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            logps = model.forward(inputs)
            loss = criterion(logps, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Valid loss: {valid_loss/len(validloader):.3f}.. "
                      f"Valid accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()

    model.class_to_idx = train_dataset.class_to_idx

    return model



def validation(model, testloader, criterion, device='cuda'):
    
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
    test_loss = 0
    accuracy = 0
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        batch_loss = criterion(logps, labels)

        test_loss += batch_loss.item()

        # Calculate accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    return test_loss, accuracy


def mk_checkpoint(model, epochs=5, input_size=25088, output_size=102, arch='vgg16', hidden_units=4096):
    checkpoint = {
        'input_size': 25088,
        'hidden_size': hidden_size,
        'output_size': 102,
        'hidden_layers': [each for each in model.features],
        'number of epochs': epochs,
        'classes to indices mapping': model.class_to_idx,
        'state_dict': model.state_dict()
    }

    return checkpoint


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a flower prediction model')
    parser.add_argument('data', help='data directory')
    parser.add_argument('--save_dir', help='save directory')
    parser.add_argument(
        '--arch', choices=['vgg13', 'vgg16'],
        help='model architecture')
    parser.add_argument('--learning_rate',
                        help='the learning rate',  type=float)
    parser.add_argument('--epochs', help='number of epochs',  type=int)
    parser.add_argument('--hidden_units', help='hidden units',  type=int)
    parser.add_argument('--gpu', help='execution on gpu', action="store_true")
    args = parser.parse_args()

    arch = args.arch if args.arch else 'vgg16'
    hidden_units = args.hidden_units if args.hidden_units else 4096
    lr = args.learning_rate if args.learning_rate else 0.003
    epochs = args.epochs if args.epochs else 5
    device = 'cuda' if args.gpu else 'cpu'

    train_dataset, valid_dataset, test_dataset = load_data(args.data)
    model, criterion, optimizer = build_model(
        arch=arch, hidden_units=hidden_units, lr=lr)
    model = train_model(model, train_dataset, valid_dataset,
                        validation, epochs, device=device)
    checkpoint = mk_checkpoint(model, arch=arch, hidden_units=hidden_units)

    if args.save_dir:
        torch.save(checkpoint, args.save_dir + 'checkpoint.pth')
    else:
        torch.save(checkpoint, 'checkpoint.pth')