import torch.nn as nn
from torchvision import models


class FaceResNet18(nn.Module):

    def __init__(self):
        super(FaceResNet18, self).__init__()
        resnet = models.resnet18(pretrained=True)
        kept_layers = list(resnet.children())[:-1]
        kept_layers.append(nn.Linear(512, 30))
        self.model = nn.Sequential(*kept_layers)

    def forward(self, x):
        x = self.model(x)
        return x


class FaceResNet50(nn.Module):

    def __iniacet__(self):
        super(FaceResNet50, self).__init__()
        resnet = models.resnet50(pretrained=True)
        kept_layers = list(resnet.children())[:-1]
        kept_layers.append(nn.Linear(2048, 30))
        self.model = nn.Sequential(*kept_layers)

    def forward(self, x):
        x = self.model(x)
        return x