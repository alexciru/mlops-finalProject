import torch.nn as nn
from torch import nn, optim
import pytorch_lightning as pl
import timm
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pytest

def build_model(pretrained=True, fine_tune=True, num_classes=43):
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
        model = timm.create_model('mobilenetv3_small_100', pretrained=True)
    else:
        print('[INFO]: Not loading pre-trained weights')
        #model = models.mobilenet_v3_small()
        model = timm.create_model('mobilenetv3_small_100', pretrained=False)
    
    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False

    # Change the final classification head.
    # Get the number of input features for the last layer
    num_ftrs = model.classifier.in_features
    # Replace the last layer with a new linear layer of size 43
    model.classifier = nn.Linear(in_features=num_ftrs, out_features=num_classes)
    return model


class MobileNetV3Lightning(pl.LightningModule):
    def __init__(self, learn_rate, num_classes=43, pretained=False):
        super(MobileNetV3Lightning, self).__init__()
        self.model = build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learn_rate)# weight_decay=0.1)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor):
        if x.ndim != 4:
            raise ValueError('Expected input to a 4D tensor')
        if x.shape[1] != 3:
            raise ValueError('Expected 3 channels input')
        if (x.shape[2] < 32 or x.shape[2] > 224):
            raise ValueError('Expected input height between 32 and 224 pixels')
        if (x.shape[3] < 32 or x.shape[3] > 224):
            raise ValueError('Expected input width between 32 and 224 pixels')
        x = self.model(x)
        return x

    # Train the model
    def training_step(self, images, labels):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(images)
        loss = self.criterion(output, labels)
        loss.backward()
        self.optimizer.step()
        self.log('train_loss', loss)
        preds = torch.argmax(F.log_softmax(output, dim=1), 1)
        return loss, preds

    # Test the model
    def test_model(self, testloader):
        self.model.eval()
        with torch.no_grad():
            correct = 0
            for (inputs, labels) in tqdm(testloader):
                outputs = self.model(inputs)
                outputs = F.log_softmax(outputs, dim=1)
                predicted = torch.argmax(outputs.data, 1)
                correct += predicted.eq(labels.data).cpu().sum()
        acc = correct / len(testloader.dataset) * 100
        return acc