import torch.nn as nn
from torch import nn, optim
import pytorch_lightning as pl
import torchvision.models as models
import torch
import torch.nn.functional as F
from tqdm import tqdm

def build_model(pretrained=True, fine_tune=True, num_classes=43):
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    else:
        print('[INFO]: Not loading pre-trained weights')
        model = models.mobilenet_v3_small()
        
        # model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    
    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False

    # Change the final classification head.
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features=num_ftrs, out_features=num_classes)

    return model


class MobileNetV3Lightning(pl.LightningModule):
    # criterion = nn.NLLLoss()
    def __init__(self, num_classes=43, pretained=False):
        super().__init__()
        self.model = build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)#, weight_decay=0.1)
        self.criterion = nn.CrossEntropyLoss()


    def forward(self, x: torch.Tensor):
        x = self.model(x)
        #x = F.log_softmax(x, dim=1)
        # breakpoint()

        return x

    def training_step(self, images, labels):
        self.model.train()
        self.optimizer.zero_grad()
        # images = images.view(images.shape[0], -1)
        output = self.model(images)
        loss = self.criterion(output, labels)
        loss.backward()
        self.optimizer.step()
        self.log('train_loss', loss)
        preds = torch.argmax(F.log_softmax(output, dim=1), 1)
        # train_running_correct += (preds == labels).sum().item()

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