import torch.nn as nn
import timm

class ModifiedMobileNetV3(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModifiedMobileNetV3, self).__init__()
        self.base_model = timm.create_model('mobilenetv3_large_100', pretrained=True)
        in_features = self.base_model.classifier.in_features
        self.base_model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.base_model(x)
        return x
