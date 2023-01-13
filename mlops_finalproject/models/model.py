import torch
import torch.nn as nn
import torchvision.models as models

# import the MobileNetV3 model
model = models.mobilenet_v3(pretrained=True)

# change the last layer to have a different number of output classes
model.classifier[1] = nn.Linear(in_features=1280, out_features=10, bias=True)

# Freeze the layers
for param in model.parameters():
    param.requiresGrad = False

# unfreeze the last layer
for param in model.classifier[1].parameters():
    param.requiresGrad = True

