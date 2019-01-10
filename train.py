# test 12/29/2018


# Imports here
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.nn import Module, Sequential 
import torch 
from torch import nn
from torch.optim import lr_scheduler
import argparse
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
import PIL
from PIL import Image
import argparse
import train_sup
#import predict_sup

# Setting Up a Parser for command line option and argument parsing 
parser = argparse.ArgumentParser(description = 'Train') 
parser.add_argument('data_dir', default = "./flower" )
parser.add_argument('--gpu', dest="gpu", action="store", default="gpu")
parser.add_argument('--arch', default = "vgg19", dest="arch", action="store", type = str )
parser.add_argument('--learning_rate', default= 0.001, type = float, action = "store" )
parser.add_argument('--epochs', dest="epochs", action="store", type=int, default= 10)
parser.add_argument('--hidden_unit', type=int, dest="hidden_unit", action="store", default=512)
parser.add_argument('--save_dir', default = "./checkpoint.pth")


# INPUT ARGUMENTS 
args = parser.parse_args() 
num_epochs = args.epochs 
learning_rate = args.learning_rate 
hidden_unit = args.hidden_unit
arch = args.arch
data_dir = args.data_dir
path = args.save_dir
gpu = args.gpu

trainloader, validloader, testloader, dataset_train  = train_sup.data_load(data_dir)
model = train_sup.CNN_model(arch, hidden_unit, gpu)
train_sup.train_nn(model, learning_rate, num_epochs, trainloader, validloader, gpu)
train_sup.save_checkpoint(model, dataset_train, arch, path, learning_rate, num_epochs)
print("Finished training")
           