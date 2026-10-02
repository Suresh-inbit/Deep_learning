import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, input_dim=784, num_classes=10):
        super(Classifier, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        # Outputs raw logits; nn.CrossEntropyLoss applies log-softmax itself
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        """
        Forward pass for the classifier.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
                or an image batch that flattens to it.
        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes).
        """
        x = x.view(x.size(0), -1)
        x = self.model(x)
        return x
