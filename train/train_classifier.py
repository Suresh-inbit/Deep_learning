"""
Train and evaluate an MLP classifier on FashionMNIST.

Usage:
    python train/train_classifier.py --epochs 10
    python train/train_classifier.py --test-only
"""
import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.classifier import Classifier
from utils import get_device, get_transform, plot_images, plot_losses, set_seed


def get_loader(train, data_dir="data", batch_size=128, num_workers=4, device="cpu"):
    return DataLoader(
        datasets.FashionMNIST(data_dir, train=train, download=True, transform=get_transform()),
        batch_size=batch_size, shuffle=train, pin_memory=(device == "cuda"), num_workers=num_workers
    )


def train_classifier(
    epochs=10,
    batch_size=128,
    lr=0.001,
    data_dir="data",
    checkpoint_dir="checkpoints",
    image_dir="images",
    num_workers=4,
    device=None
):
    """
    Train a simple classifier on the FashionMNIST dataset.
    """
    device = get_device(device)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    dataloader = get_loader(True, data_dir, batch_size, num_workers, device)
    dataset = dataloader.dataset
    sample_count = 25
    plot_images([dataset[i][0] for i in range(sample_count)],
                os.path.join(image_dir, "classifier_samples.png"),
                titles=[dataset.classes[dataset[i][1]] for i in range(sample_count)])

    input_dim = dataset[0][0].numel()
    num_classes = len(dataset.classes)
    print(f"Input dimension: {input_dim}, Number of classes: {num_classes}")

    classifier = Classifier(input_dim=input_dim, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=lr)
    acc = Accuracy(task="multiclass", num_classes=num_classes).to(device)

    losses = []
    for epoch in range(epochs):
        classifier.train()
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = classifier(images)
            loss = criterion(outputs, labels)
            acc.update(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(dataloader)}], "
                      f"Loss: {loss.item():.4f}")

        print(f"Epoch [{epoch + 1}/{epochs}] training accuracy: {acc.compute().item():.4f}")
        acc.reset()

    # Save the classifier with the settings needed to rebuild it
    checkpoint_path = os.path.join(checkpoint_dir, "classifier.pth")
    torch.save({
        "state_dict": classifier.state_dict(),
        "input_dim": input_dim,
        "num_classes": num_classes,
    }, checkpoint_path)
    print(f"Model saved to '{checkpoint_path}'")

    plot_losses({"Loss": losses}, os.path.join(image_dir, "classifier_loss_curve.png"),
                "Classifier Training Loss")
    print("Training complete.")
    return classifier


def test_classifier(
    batch_size=128,
    data_dir="data",
    checkpoint_dir="checkpoints",
    image_dir="images",
    num_workers=4,
    device=None
):
    """
    Test the classifier on the FashionMNIST test set.
    """
    device = get_device(device)
    os.makedirs(image_dir, exist_ok=True)
    test_loader = get_loader(False, data_dir, batch_size, num_workers, device)
    classes = test_loader.dataset.classes

    checkpoint = torch.load(os.path.join(checkpoint_dir, "classifier.pth"), map_location=device)
    classifier = Classifier(input_dim=checkpoint["input_dim"],
                            num_classes=checkpoint["num_classes"]).to(device)
    classifier.load_state_dict(checkpoint["state_dict"])
    classifier.eval()
    acc = Accuracy(task="multiclass", num_classes=checkpoint["num_classes"]).to(device)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = classifier(images)
            acc.update(outputs, labels)
    print(f"Test accuracy: {acc.compute().item():.4f}")

    # Plot predictions for the last batch
    preds = outputs.argmax(dim=1)
    n = min(25, images.size(0))
    output_path = os.path.join(image_dir, "classifier_predictions.png")
    plot_images(images[:n], output_path,
                titles=[f"{classes[p]} ({classes[t]})" for p, t in zip(preds[:n].tolist(), labels[:n].tolist())])
    print(f"Predictions (predicted (true)) saved to '{output_path}'")


def parse_args():
    parser = argparse.ArgumentParser(description="Train an MLP classifier on FashionMNIST.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--image-dir", default="images")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None, help="cpu or cuda (default: cuda if available)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-only", action="store_true",
                        help="skip training and evaluate the saved checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    common = dict(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        image_dir=args.image_dir,
        num_workers=args.num_workers,
        device=args.device,
    )
    if not args.test_only:
        train_classifier(epochs=args.epochs, lr=args.lr, **common)
    test_classifier(**common)


if __name__ == "__main__":
    main()
