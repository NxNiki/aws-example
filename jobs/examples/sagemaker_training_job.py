import json
import os

import numpy as np
import smdebug.pytorch as smd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import ImageFile
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

import argparse


def test(model, test_loader, criterion, hook=None):
    """
    Complete this function that can take a model and a testing data loader and will get the test accuracy/loss of the
    model. Remember to include any debugging/profiling hooks that you might need
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model.eval()
    # ===================================================#
    # 3. Set the SMDebug hook for the validation phase. #
    # ===================================================#
    if hook:
        hook.set_mode(smd.modes.EVAL)

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )


def train(model, train_loader, epochs, criterion, optimizer, hook=None):
    """
    Complete this function that can take a model and data loaders for training and will get train the model
    Remember to include any debugging/profiling hooks that you might need

    :param model:
    :param train_loader:
    :param epochs:
    :param criterion:
    :param optimizer:
    :param hook:
    :return:
    """

    model.train()

    if hook:
        hook.set_mode(smd.modes.TRAIN)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            # loss = F.nll_loss(output, target)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )


def net():
    """
    Complete this function that initializes your model
    Remember to use a pretrained model
    """

    model = models.resnet18(weights="DEFAULT")
    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 133))

    return model


def create_data_loaders(data_train, data_test, batch_size_train, batch_size_test):
    """
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    """

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize to 224x224
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet normalization
        ]
    )

    train_dataset = datasets.ImageFolder(root=data_train, transform=transform)
    test_dataset = datasets.ImageFolder(root=data_test, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=int(batch_size_train.strip('"')), shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=int(batch_size_test.strip('"')), shuffle=False, num_workers=2)

    return train_loader, test_loader


def main(args):
    train_loader, test_loader = create_data_loaders(
        args.data_train, args.data_test, args.batch_size, args.test_batch_size
    )

    """
    TODO: Initialize a model by calling the net function
    """
    model = net()

    """
    TODO: Create your loss and optimizer
    """
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)

    if os.path.exists("/opt/ml/input/config/debughookconfig.json"):
        hook = smd.Hook.create_from_json_file()
        hook.register_hook(model)
    else:
        hook = None

    """
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    """
    train(model, train_loader, args.epochs, loss_criterion, optimizer, hook)

    """
    TODO: Test the model to see its accuracy
    """
    test(model, test_loader, loss_criterion, hook)

    """
    TODO: Save the trained model
    """
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))


if __name__ == "__main__":
    """
    TODO: Specify any training args that you might need
    """
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--batch-size",
        type=str,
        default="64",
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=str,
        default="1000",
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument("--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)")
    parser.add_argument("--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)")

    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--data-test", type=str, default=os.environ["SM_CHANNEL_TEST"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    args = parser.parse_args()

    main(args)
