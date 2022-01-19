import torch.nn as nn
from torchvision import models


class TunedResNet18(nn.Module):

    def __init__(self):
        super(TunedResNet18, self).__init__()
        self.model = models.ResNet18(pretrained=True)