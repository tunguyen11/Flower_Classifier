import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.nn import Module, Sequential 
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
import PIL
from PIL import Image
import argparse


# data Loader 
def data_load(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir  = data_dir + '/test'
# TODO: Define your transforms for the training, validation, and testing sets
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(45),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                [0.229, 0.224, 0.225])
        ]),
    }

# TODO: Load the datasets with ImageFolder
    dataset_train = datasets.ImageFolder(train_dir, transform = data_transforms['train'])
    dataset_test = datasets.ImageFolder(test_dir, transform = data_transforms['test'])
    dataset_valid = datasets.ImageFolder(valid_dir, transform = data_transforms['valid'])
                     
# TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader =   torch.utils.data.DataLoader(dataset_train, batch_size = 32, shuffle = True)
    validloader =  torch.utils.data.DataLoader(dataset_valid, batch_size = 32, shuffle = False)
    testloader =    torch.utils.data.DataLoader(dataset_test, batch_size = 32, shuffle = False)
    return trainloader , validloader, testloader, dataset_train

def CNN_model(arch, hidden_unit, gpu):
    
    if arch == "vgg16":
        model = models.vgg16(pretrained = True) 
    elif arch == "vgg19":
        model = models.vgg19(pretrained = True)
    else: 
        print("the model is not available")
    for param in model.parameters():
        param.requires_grad = False 
    classifier = Sequential(
                            nn.Linear(in_features=25088, out_features=hidden_unit, bias=True),
                            nn.ReLU(),
                            nn.Dropout(p=0.3),
                            nn.Linear(in_features = hidden_unit, out_features = 102 , bias=True), 
                            nn.LogSoftmax(dim = 1)
                        )
    model.classifier = classifier
    if torch.cuda.is_available() and gpu == "gpu": 
        model.cuda()
    return model

# train the network 

def train_nn(model,learning_rate, num_epochs, trainloader, validloader, gpu):
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr = learning_rate)
    model.train()
    
    for epoch in range(num_epochs):
        train_loss = 0
        train_correct = 0 
        train_total = 0 
        for i, (images, labels) in enumerate(trainloader):
            if torch.cuda.is_available() and gpu == "gpu":
                images = images.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()
            # Forward and backward passes
            outputs  = model(images)
            ps = torch.exp(outputs).data
            loss = criterion(outputs, labels) 
            loss.backward()
            optimizer.step()
            probs, predictions = torch.max(ps, dim =  1) 
            
            train_loss += loss
            train_correct += (predictions ==labels.data).sum().item()
            train_total += labels.size(0) 
            
            if (i+1)% 20 ==0: 
                model.eval()
                valid_loss = 0 
                valid_correct = 0
                valid_total = 0 
                for j, (images2, labels2) in enumerate(validloader): 
                    optimizer.zero_grad()
                    if torch.cuda.is_available() and gpu == "gpu":
                        model.to('cuda:0')
                        images2 = images2.to('cuda:0')
                        labels2 = labels2.to('cuda:0')
                    with torch.no_grad():
                        outputs2 = model(images2)
                        ps2 = torch.exp(outputs2).data
                        probs2, predictions2 = torch.max(ps2, dim = 1)
                        loss2 = criterion(outputs2, labels2)
                        valid_loss += loss2 
                        valid_correct += (predictions2 == labels2.data).sum().item()
                        valid_total += labels2.size(0)
                
                print("Epoch {}/{}: ".format(epoch+1, num_epochs),
                        "Step {}/{}:".format(i+1, len(trainloader)),
                      "Training Loss {}". format(train_loss/train_total),
                        "Valid Loss {}". format(valid_loss/valid_total),
                        "Valid Accuracy: {}".format(valid_correct/valid_total))
        #print("model in the training nw")
        #print(model.state_dict())

              
def save_checkpoint(model, dataset_train, arch, path, learning_rate, epochs):
  
    #print("model in checkpoint")
    #print(model.state_dict())
    checkpoint = {'model_state': model.state_dict(),
              #'criterion_state': criterion.state_dict(),
              #'optimizer_state': optimizer.state_dict(),
                'arch': arch, 
                'class_to_idx': dataset_train.class_to_idx ,
               #'hidden_layer': hidden_layer, 
              'arch' : arch, 
               'lr' : learning_rate, 
               'epochs': epochs
              }
    checkpoint = torch.save(checkpoint, path)
    
    
                   
            
            
            
    
    
    
     
       
        


    
