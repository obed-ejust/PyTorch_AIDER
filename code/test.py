import torch
from torch import nn
from torchvision import models
from my_models import *
from emergencyNet import ACFFModel

"""
saved_model = '../results/model.pth'
model = torch.load(saved_model)
model.eval()
"""
# model2 = select_model("ResNet50", 5)
model = ACFFModel(5)

print(model)
