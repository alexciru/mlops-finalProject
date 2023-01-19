import torch.nn as nn
from torch import nn, optim
import pytorch_lightning as pl
import timm
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torchmetrics
import numpy as np
import timm


class MobileNetV3Lightning(pl.LightningModule):
    def __init__(self, num_classes=43, pretained=False):
        super(MobileNetV3Lightning, self).__init__()
        self.model = self.build_model()
        # self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)#, weight_decay=0.1)
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x: torch.Tensor):
        if x.ndim != 4:
            raise ValueError("Expected input to a 4D tensor")
        if x.shape[1] != 3:
            raise ValueError("Expected 3 channels input")
        if x.shape[2] < 32 or x.shape[2] > 224:
            raise ValueError("Expected input height between 32 and 224 pixels")
        if x.shape[3] < 32 or x.shape[3] > 224:
            raise ValueError("Expected input width between 32 and 224 pixels")
        x = self.model(x)
        return x

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        return optimizer

    def training_step(self, batch, batch_idx):
        # self.model.train()

        # self.optimizer.zero_grad()
        inputs, labels = batch
        # breakpoint()

        output = self.model(inputs)
        loss = self.criterion(output, labels)
        # loss.backward()
        # self.optimizer.step()
        preds = torch.argmax(F.log_softmax(output, dim=1), 1)
        # train_running_correct += (preds == labels).sum().item()
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)

        outputs = F.log_softmax(outputs, dim=1)
        predicted = torch.argmax(outputs.data, 1)

        acc = self.accuracy(outputs, labels)
        # self.log('val_accuracy', acc, on_step=True, on_epoch=True)

        # return predicted, labels
        return acc
        # correct += predicted.eq(labels.data).cpu().sum()

    def validation_epoch_end(self, validation_step_outputs):
        self.log("val_accuracy", np.mean(validation_step_outputs) * 100)

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        imgs, label = batch
        outputs = self.model(imgs)
        outputs = F.log_softmax(outputs, dim=1)
        predicted = torch.argmax(outputs.data, 1)

        return predicted

    def build_model(pretrained=True, fine_tune=True, num_classes=43):
        if pretrained:
            print("[INFO]: Loading pre-trained weights")
            # model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
            model = timm.create_model("mobilenetv3_small_100", pretrained=True)
        else:
            print("[INFO]: Not loading pre-trained weights")
            # model = models.mobilenet_v3_small()
            model = timm.create_model("mobilenetv3_small_100", pretrained=False)

            # model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Large_Weights.DEFAULT)

        if fine_tune:
            print("[INFO]: Fine-tuning all layers...")
            for params in model.parameters():
                params.requires_grad = True
        elif not fine_tune:
            print("[INFO]: Freezing hidden layers...")
            for params in model.parameters():
                params.requires_grad = False

        # Change the final classification head.
        # num_ftrs = model.classifier[-1].in_features
        # model.classifier[-1] = nn.Linear(in_features=num_ftrs, out_features=num_classes)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(in_features=num_ftrs, out_features=num_classes)

        return model

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
