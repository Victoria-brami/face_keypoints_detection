import torch
import os
from torch.utils.data import DataLoader
from src.training_parser import parser
from src.trainer import train, test
from src.dataset import FaceDataset
from src.utils import get_optimizer, get_augmentation, get_model
from torch.utils.tensorboard import SummaryWriter



def train_on_epochs(model, loss, datasets, optimizer, parameters, writer):

    # prepare loaders
    train_dataset = datasets["train"]
    train_loader = DataLoader(train_dataset, batch_size=parameters["batch_size"],
                                shuffle=True, num_workers=2)
    test_dataset = datasets["test"]
    test_loader = DataLoader(test_dataset, batch_size=parameters["batch_size"],
                              shuffle=True, num_workers=2)

    logpath = os.path.join(parameters["checkpoint_dir"], "training.log")
    with open(logpath, "w") as logfile:
        for epoch in range(1, parameters["n_epochs"]+1):
            dict_loss = train(model, loss, optimizer, train_loader, parameters["train_device"])
            val_loss = test(model, loss, optimizer, test_loader, parameters["test_device"])

            writer.add_scalars('Loss', {'Train': dict_loss["MSE"], 'Val': val_loss["MSE"]})
            print(f"Epoch {epoch}:  Train loss: {dict_loss} and Val loss: {val_loss}")
            print(f"Epoch {epoch}:  Train loss: {dict_loss} and Val loss: {val_loss}", file=logfile)

            if ((epoch % parameters["snapshot"]) == 0) or (epoch == parameters["n_epochs"]):
                checkpoint_path = os.path.join(parameters["checkpoint_dir"],
                                               'checkpoint_{:04d}.pth.tar'.format(epoch))
                print('Saving checkpoint {}'.format(checkpoint_path))
                torch.save(model.state_dict(), checkpoint_path)
        writer.flush()


def main():

    # Add options
    parameters = parser()

    # logging tensorboard
    writer = SummaryWriter(log_dir=parameters["writer_folder"])

    # Define the model
    model = get_model(parameters)

    # Get augmentation
    train_trsfm = get_augmentation('train', parameters)
    test_trsfm = get_augmentation('test', parameters)

    # Build the datasets
    train_dataset = FaceDataset(transform=train_trsfm, mode='train', datapath='../data')
    test_dataset = FaceDataset(transform=test_trsfm, mode='test', datapath='../data')

    datasets = {'train': train_dataset, 'test': test_dataset}
    train_loader = DataLoader(train_dataset, batch_size=parameters["batch_size"],
                             shuffle=True, num_workers=parameters["num_workers"])
    test_loader = DataLoader(test_dataset, batch_size=parameters["batch_size"],
                             shuffle=True, num_workers=parameters["num_workers"])

    # define the optimizer
    optimizer = get_optimizer(model, parameters)

    # Define the loss
    loss = torch.nn.MSELoss(reduction='mean')

    # Training loop
    train_on_epochs(model, loss, datasets, optimizer, parameters, writer)

    writer.close()


if __name__ == '__main__':
    main()

