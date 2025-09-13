import argparse
import json
import os

import numpy as np
import smdebug.pytorch as smd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import ImageFile
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, models, transforms

# Print PyTorch version and CUDA info for debugging
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")

# Handle mixed precision training with compatibility for different PyTorch versions
amp_module = None
try:
    from torch import amp

    amp_module = amp
    print("Using torch.amp for mixed precision training")
except ImportError:
    try:
        from torch.cuda import amp

        amp_module = amp
        print("Using torch.cuda.amp for mixed precision training")
    except ImportError:
        print("Mixed precision training not available, falling back to full precision")


# Check if we can actually use mixed precision
def is_amp_available():
    if amp_module is None:
        return False
    try:
        # Test if GradScaler is available
        _ = amp_module.GradScaler()
        return True
    except (AttributeError, TypeError):
        return False


# Initialize distributed training
local_rank = 0
if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    print(
        f"Initialized distributed training: rank {dist.get_rank()} / {dist.get_world_size()}, local_rank: {local_rank}"
    )
    print(f"CUDA available: {torch.cuda.is_available()}, CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}, Device name: {torch.cuda.get_device_name()}")
else:
    print("Single-node training")

# Determine device once at module level
DEVICE = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

ImageFile.LOAD_TRUNCATED_IMAGES = True


def test(model, test_loader, criterion, hook=None):
    """
    Complete this function that can take a model and a testing data loader and will get the test accuracy/loss of the
    model. Remember to include any debugging/profiling hooks that you might need
    """

    # Model should already be on correct device from main function
    print(f"Testing model on device: {next(model.parameters()).device}")
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
            data = data.to(DEVICE)
            target = target.to(DEVICE)
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

    # Model should already be on correct device from main function
    print(f"Training model on device: {next(model.parameters()).device}")
    model.train()

    if hook:
        hook.set_mode(smd.modes.TRAIN)

    # Initialize mixed precision training if available
    scaler = None
    amp_available = is_amp_available()
    print(f"Mixed precision availability check: {amp_available}")
    if amp_available:
        try:
            scaler = amp_module.GradScaler()
            print("Mixed precision training enabled with GradScaler")
        except (AttributeError, TypeError) as e:
            print(f"GradScaler not available ({e}), falling back to full precision")
            amp_available = False
        except Exception as e:
            print(f"Unexpected error initializing GradScaler ({e}), falling back to full precision")
            amp_available = False

    for epoch in range(epochs):
        # Set epoch for distributed sampler
        if (
            "WORLD_SIZE" in os.environ
            and int(os.environ["WORLD_SIZE"]) > 1
            and hasattr(train_loader.sampler, "set_epoch")
        ):
            train_loader.sampler.set_epoch(epoch)

        samples_processed = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            optimizer.zero_grad()

            # Use mixed precision if available, otherwise use full precision
            if amp_available and scaler is not None:
                try:
                    with amp_module.autocast():
                        output = model(data)
                        loss = criterion(output, target)

                    scaler.scale(loss).backward()  # Scale loss before backward()
                    scaler.step(optimizer)  # Unscale gradients and call optimizer.step()
                    scaler.update()  # Update the scaler for the next iteration
                except Exception as e:
                    print(f"Mixed precision training failed ({e}), falling back to full precision")
                    amp_available = False
                    # Fall back to full precision for this batch
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
            else:
                # Full precision training
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            samples_processed += len(data)
            if batch_idx % 100 == 0 or batch_idx == len(train_loader) - 1:
                print(
                    f"Train Epoch: {epoch} [{samples_processed}/{len(train_loader.dataset)} ({100.0 * (batch_idx+1) / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
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

    num_workers = min(os.cpu_count(), 8)
    train_dataset = datasets.ImageFolder(root=data_train, transform=transform)
    test_dataset = datasets.ImageFolder(root=data_test, transform=transform)

    # Set up distributed sampling for training
    train_sampler = None
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        train_sampler = DistributedSampler(
            train_dataset, shuffle=True, num_replicas=dist.get_world_size(), rank=dist.get_rank()
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,  # Important for DDP to avoid uneven batch sizes
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size_test, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader


def main(args):
    try:
        train_loader, test_loader = create_data_loaders(
            args.data_train, args.data_test, args.batch_size, args.test_batch_size
        )

        """
        Initialize a model by calling the net function
        """
        model = net()
        print(f"Model created successfully: {type(model)}")
        print(f"Model fc layer: {model.fc}")
        print(f"Model fc layer type: {type(model.fc)}")

        # Move model to device BEFORE DDP wrapping
        model = model.to(DEVICE)
        print(f"Model created and moved to device: {next(model.parameters()).device}")

        # Wrap model with DistributedDataParallel if using distributed training
        if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
            from torch.nn.parallel import DistributedDataParallel as DDP

            model = DDP(model, device_ids=[local_rank])
            print(f"Model wrapped with DDP, device_ids: {[local_rank]}")
            print(f"DDP model structure - module.fc exists: {hasattr(model.module, 'fc')}")
            print(f"DDP model structure - fc layer: {model.module.fc}")

        """
        Create loss and optimizer
        """
        loss_criterion = nn.CrossEntropyLoss()

        # Scale learning rate for distributed training
        lr = args.lr
        if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
            # Scale LR by sqrt of world size for more stable training
            lr = lr * (dist.get_world_size() ** 0.5)
            print(
                f"Scaled learning rate from {args.lr} to {lr} for distributed training (world_size={dist.get_world_size()})"
            )

        # Access fc layer correctly whether using DDP or not
        if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
            print(f"Creating optimizer for DDP model using model.module.fc")
            optimizer = optim.Adam(model.module.fc.parameters(), lr=lr)
        else:
            print(f"Creating optimizer for single model using model.fc")
            optimizer = optim.Adam(model.fc.parameters(), lr=lr)

        if os.path.exists("/opt/ml/input/config/debughookconfig.json"):
            hook = smd.Hook.create_from_json_file()
            hook.register_hook(model)
        else:
            hook = None

        """
        Call the train function to start training your model
        Remember that you will need to set up a way to get training data from S3
        """
        train(model, train_loader, args.epochs, loss_criterion, optimizer, hook)

        """
        Test the model to see its accuracy
        """
        test(model, test_loader, loss_criterion, hook)

        """
        Save the trained model
        """
        # Only save model on main process to avoid conflicts
        if not "WORLD_SIZE" in os.environ or int(os.environ["WORLD_SIZE"]) <= 1 or dist.get_rank() == 0:
            if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
                # Save the underlying model, not the DDP wrapper
                torch.save(model.module.state_dict(), os.path.join(args.model_dir, "model.pth"))
            else:
                torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    """
    Specify any training args that you might need
    """
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
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

    parser.add_argument("--hosts", type=str, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--data-test", type=str, default=os.environ["SM_CHANNEL_TEST"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    args = parser.parse_args()

    main(args)
