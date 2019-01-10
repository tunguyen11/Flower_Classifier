import numpy as np 
import seaborn as sns 
from matplotlib import pyplot as plt  
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
import train_sup
hidden_unit = 512
# load the checkpoint
def load_checkpoint(gpu, path): 
    cp = torch.load(path)
    arch = cp['arch']
    lr = cp['lr']
    state_dict = cp['model_state']
    class_to_idx = cp['class_to_idx']
    model = train_sup.CNN_model(arch, hidden_unit, gpu )
    model.load_state_dict(state_dict)
    #optimizer = nn.optim.Adams(model.classifier.parameters(), lr = lr )
    #optimizer.load_state_dict(cp['optimizer'])
    return  model,  arch, class_to_idx

def map_idx_label(category_names, class_to_idx):
    import json
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
        # mapping the index to the category name 
    idx_to_class = {key: val for val, key in class_to_idx.items() }
    idx_to_label = {key: cat_to_name[str(val)] for key, val in  idx_to_class.items() }
    return  idx_to_label, idx_to_class, cat_to_name

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    from PIL import Image
    img = Image.open(image_path)
     # Normalize
    transform_f = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                [0.229, 0.224, 0.225])
        ])
    img = transform_f(img)
    return img

def imshow(image_path, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = process_image(image_path)
    image = np.array(image)
    image = image.transpose((1, 2, 0))
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    return ax


def predict(image_path, model, topk, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    image = process_image(image_path)
    
    # add the batch size of 1 for one single image 
    image_input = image.unsqueeze(0)
    image_input = image_input.float()
    if torch.cuda.is_available() and gpu == "gpu":
        model.to('cuda:0')
    if gpu == 'gpu':
        with torch.no_grad():
            output = model(image_input.cuda())
    else:
        with torch.no_grad():
            output=model(image_input)
   
    ps = torch.exp(output)
    top_probs, top_idxs = torch.topk(ps, topk)
    top_probs = top_probs.detach().tolist()[0]
    top_idxs = top_idxs.detach().tolist()[0]
    return top_probs, top_idxs 
    

