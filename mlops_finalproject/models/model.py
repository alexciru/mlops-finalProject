import torch.nn as nn
from torch import nn, optim
import pytorch_lightning as pl
import torchvision.models as models
import torch
import torch.nn.functional as F


def build_model(pretrained=False, fine_tune=True, num_classes=43):
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
    else:
        print('[INFO]: Not loading pre-trained weights')

    # model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False

    # Change the final classification head.
    model.classifier[3] = nn.Linear(in_features=1024, out_features=num_classes)
    return model


class MobileNetV3Lightning(pl.LightningModule):
    # criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()
    def __init__(self, num_classes=43, pretained=False):
        super().__init__()
        self.model = build_model()

    def forward(self, x: torch.Tensor):
        x = self.model(x)
        # x = F.log_softmax(x, dim=1)
        # breakpoint()

        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch
        # images = images.view(images.shape[0], -1)
        output = self(images)
        loss = self.criterion(output, labels)
        self.log('train_loss', loss)

        _, preds = torch.max(output.data, 1)

        # train_running_correct += (preds == labels).sum().item()

        return loss

    def configure_optimizers(self):
        # params =
        optimizer = optim.Adam(self.parameters(), lr=0.01)
        return optimizer

    # def validation_step(self, data):

    #     return 
    
    def test_step(self, batch, batch_idx):
        images, labels = batch

        # resize images
        # images = images.view(images.shape[0], -1)

        # Old stuff
        # logps = self(images)
        # # logps = mymodel.forward(images)
        # ps = torch.exp(logps)


        # # Take max from the probs
        # top_p, top_class = ps.topk(1, dim=1)

        # # Compare with labels
        # equals = top_class == labels.view(*top_class.shape)

        # # mean
        # accuracy = torch.mean(equals.type(torch.FloatTensor))
        # self.log('test_accuracy', accuracy)
        # return accuracy

        # New stuff
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        _, preds = torch.max(outputs.data, 1)
        # acc = torch.mean((torch.argmax(preds, dim=1) == labels).float())
        acc = torch.mean((preds == labels).float())
        self.log("test_acc", acc)
        
