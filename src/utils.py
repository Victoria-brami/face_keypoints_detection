import torch
import torch.optim as optim
from src.model import FaceResNet18, FaceResNet50
from torchvision import transforms
from skimage import io, transform

class Resize(object):
    def __init__(self, output_size=224):
        self.output_size=output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        input_size = image.shape[0]
        img = transform.resize(image, (self.output_size, self.output_size))
        label *= self.output_size / input_size

        sample = {'image': image, 'label': label}

        return sample

class ToTensor(object):

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        image = image.repeat(3, 1, 1) # COnvert to 3 channels images for ResNet input
        sample = {'image': image,
                'label': torch.from_numpy(label)}
        return sample


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
            trsfm_list.append(RandomHorizontalFlip())
    trsfm_list.append(Resize(224))
    trsfm_list.append(ToTensor())
    return transforms.Compose(trsfm_list)