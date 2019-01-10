import numpy
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
import predict_sup
import train_sup

# Setting Up a Parser for command line option and argument parsing 
parser = argparse.ArgumentParser(description='predict')
parser.add_argument('image_path', default='./flowers/test/1/image_06752.jpg', type=str)
parser.add_argument('checkpoint', default='./checkpoint.pth', type=str)
parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
parser.add_argument('--category_names', dest="category_names", action="store", default='./cat_to_name.json')
parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")

args = parser.parse_args()
image_path = args.image_path
checkpoint = args.checkpoint   
category_names = args.category_names                                                                                            
topk = args.top_k
gpu= args.gpu
                                 
# load the checkpoint 
hidden_unit = 512
print(type(image_path))
print(image_path)

model, arch, class_to_idx = predict_sup.load_checkpoint(gpu, checkpoint)
# ## Class and label mapping 
idx_to_label, idx_to_class, cat_to_name= predict_sup.map_idx_label(category_names, class_to_idx) 

# TODO: Implement the code to predict the class from an image file
top_probs, top_idxs = predict_sup.predict(image_path, model, topk, gpu)
top_classes = [idx_to_class[i] for i in top_idxs]
top_flowers = [idx_to_label[i] for i in top_idxs]

flower_num = image_path.split('/')[-2]
flower_name = cat_to_name[flower_num]
print("The true name of flower is {}".format(flower_name))
print("The most likely flower type that the model predicts is {} and it's associated probability is {}".format(top_flowers[0], top_probs[0]))
print ("The top {} predicted flower types of the image is {}".format(topk, top_flowers))
print ("The top {} predicted classes of the image is {}".format(topk, top_classes))
print ("The top predicted probabilities correspondingly is {}".format(top_probs))
# print the predicted flowers

