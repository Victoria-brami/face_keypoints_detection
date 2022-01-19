import torch
from tqdm import tqdm
from src.loss import compute_loss

def train_or_test(model, loss, optimizer, iterator, device, mode):
    assert mode in ['train', 'test']
    if mode == "train":
        model.train()
        grad_env = torch.enable_grad
    elif mode == "test":
        model.eval()
        grad_env = torch.no_grad
    else:
        raise ValueError("This mode is not recognized.")

    loss_dict = {"MSE": 0}

    with grad_env():
        for i, batch in tqdm(enumerate(iterator), desc="Computing Batch ..."):
            # we ensure the data is on the good device
            batch = {key: value.to(device) for key, value in batch.items()}

            if mode =='train':
                optimizer.zero_grad()

            # Compute the model's output
            image, label = batch['image'], batch['label']
            ouptut = model(image)

            loss_value = loss(ouptut, label)
            loss_dict['MSE'] += loss_value

            if mode == "train":
                # backward pass
                loss_value.backward()
                # update the weights
                optimizer.step()

    return loss_dict


def train(model, loss, optimizer, iterator, device):
    return train_or_test(model, loss, optimizer, iterator, device, mode="train")

def test(model, loss, optimizer, iterator, device):
    return train_or_test(model, loss, optimizer, iterator, device, mode="test")

