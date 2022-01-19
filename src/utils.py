import torch
import torch.optim as optim
from src.model import FaceResNet18, FaceResNet50
from torchvision import transforms


def get_model(parameters):
    if parameters["modelname"] == 'resnet18':
        model = FaceResNet18()
    else:
        model = FaceResNet50()

    if parameters["init_from_checkpoint"] != ' ':
        chkpt = torch.load(parameters["init_from_checkpoint"], map=parameters["device"])
        model.load_state_dict(chkpt)
    return model

def get_optimizer(model, parameters):
    if parameters["optimizer"].lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=parameters["lr"], weight_decay=parameters["wd"])
    if parameters["optimizer"].lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=parameters["lr"], weight_decay=parameters["wd"])
    if parameters["optimizer"].lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              momentum=parameters["momentum"],
                              lr=parameters["lr"],
                              weight_decay=parameters["wd"])
    return optimizer

def get_augmentation(mode, parameters):
    trsfm_list = []
    if mode == 'train':
        if "flip" in parameters["aug"]:
            trsfm_list.append(transforms.RandomHorizontalFlip())
    trsfm_list.append(transforms.Resize(224))
    trsfm_list.append(transforms.ToTensor())
    return transforms.Compose(trsfm_list)