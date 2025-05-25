import models.classifier as Classifier
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import sys
from torchmetrics import Accuracy
import torchvision.utils as vutils
def classifier(
    epochs=10,
    batch_size=128,
    lr=0.001
    
):
    """
    Train a simple classifier on MNIST dataset.
    """
    device =  "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("images", exist_ok=True)

    # Data loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(),
        transforms.Normalize([0.5], [0.5]),
        # transforms.Resize((28, 28)),  # Resize to match input dimension
    ])
    
    dataloader = DataLoader(
        datasets.FashionMNIST("fdata", train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4
    )
    plt.imshow(dataloader.dataset[0][0].numpy().squeeze(), cmap='gray')
    plt.axis('off')
    plt.show()
    # Models
    input_dim = 784  # Get input dimension from the first batch
    num_classes = len(dataloader.dataset.classes)  # Get number of classes from dataset
    print(f"Input dimension: {input_dim}, Number of classes: {num_classes}")


    # Initialize the classifier
    classifier = Classifier.Classifier(input_dim=input_dim, num_classes=num_classes).to(device)
    # classifier.load_state_dict(torch.load('classifier.pth', map_location=device))  # Load pre-trained model if available
    # Loss and optimizernch
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=lr)
    acc = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    # Training loop
    losses = []
    acc_values = []
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            # Calculate accuracy
            acc_value = acc(outputs, labels)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(dataloader)}], \
                         Loss: {loss.item():.4f}, accuracy: {acc_value.item():.4f}')

        # Save model checkpoint
    print(labels.shape)
    torch.save(classifier.state_dict(), 'classifier.pth')
    plt.plot(losses, label='Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()
    print(f'Model saved')
    print("Training complete.")
    return classifier
def test_classifier():
    """
    Test the classifier on the FashionMNIST dataset.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(),
        transforms.Normalize([0.5], [0.5]),
        transforms.Resize((28, 28)),  # Resize to match input dimension
    ])
    
    test_loader = DataLoader(
        datasets.FashionMNIST("fdata", train=False, download=True, transform=transform),
        batch_size=128, shuffle=False, pin_memory=True, num_workers=4
    )
    
    classifier = Classifier.Classifier(input_dim=784, num_classes=10).to(device)
    classifier.load_state_dict(torch.load('classifier.pth', map_location=device))
    
    classifier.eval()
    acc = Accuracy(task="multiclass", num_classes=10).to(device)
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = classifier(images)
            acc_value = acc(outputs, labels)
            print(f'Test accuracy: {acc_value.item():.4f}')
    print(acc.compute())
    plot_images(images, labels, num_images=25)

def plot_images(images, labels, num_images=25):
    """
    Plot a grid of images with their labels.
    """
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(images[i].cpu().numpy().squeeze(), cmap='gray')
        plt.title(f'Label: {labels[i].item()}')
        plt.axis('off')
    plt.show()
if __name__ == "__main__":
    model = classifier(epochs=10, batch_size=128, lr =0.001)
    # test_classifier()
    