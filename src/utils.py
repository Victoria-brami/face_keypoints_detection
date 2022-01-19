import torch
import torch.optim as optim
from torchvision import transforms


def get_model(parameters):
    model = None
    if parameters["init_from_checkpoint"] is not None:
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
            trsfm_list.append(RandomHorizontalFlip())
    trsfm_list.append(transforms.ToTensor())
    return transforms.Compose(trsfm_list)