import torch
import torch.nn as nn

class CNN(nn.Module):
    """
    A simple CNN for image classification.

    Input:  (B, C, H, W)
    Output: (B, num_classes)
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 10, hidden_dim: int = 128, kernel_size: int = 3, stride: int = 1, padding: int = 1, conv1_out_channels: int = 32, conv2_out_channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, conv1_out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(conv1_out_channels, conv2_out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.fc1 = nn.Linear(conv2_out_channels * 8 * 8, hidden_dim)  # Assuming input images are 32x32
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pool(torch.relu(self.conv1(x)))
        out = self.pool(torch.relu(self.conv2(out)))
        out = out.view(out.size(0), -1)  # Flatten
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out