from __future__ import annotations

import os
from typing import Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    """Custom Dataset for loading data and targets.

    Parameters
    ----------
    data : array-like
        The data samples.
    targets : array-like
        The target labels.
    transform : callable, optional
        A function/transform to apply to the data.
    """

    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.targets[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


class SimpleClassifier(nn.Module):
    """A simple Convolutional Neural Network (CNN) for classification.

    Methods
    -------
    forward(x)
        Defines the forward pass of the model.
    """

    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_classifier(test_set: Tuple,
                     train_set: Tuple,
                     device: torch.device = torch.device(
                         'cuda' if torch.cuda.is_available() else 'cpu'),
                     batch_size: int = 64,
                     learning_rate: float = 0.001,
                     num_epochs: int = 5) -> Tuple[nn.Module, float]:
    """Train a simple classifier on the provided dataset.

    Parameters
    ----------
    test_set : Tuple
        The test dataset as a tuple (data, targets).
    train_set : Tuple
        The training dataset as a tuple (data, targets).
    device : torch.device, optional
        The device to run the training on (default is CUDA if available).
    batch_size : int, optional
        The batch size for training (default is 64).
    learning_rate : float, optional
        The learning rate for the optimizer (default is 0.001).
    num_epochs : int, optional
        The number of epochs to train for (default is 5).

    Returns
    -------
    model : nn.Module
        The trained model.
    accuracy : float
        The accuracy of the model on the test set.
    """
    train_dataset = CustomDataset(train_set[0], train_set[1])
    test_dataset = CustomDataset(test_set[0], test_set[1])

    # Create data loaders
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = SimpleClassifier().to(device).to(train_set[0][0].dtype)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create directory for saving model states
    os.makedirs('classifier_states', exist_ok=True)

    # Load the latest model state if it exists
    start_epoch = 0
    for epoch in range(num_epochs, 0, -1):
        model_path = f'classifier_states/model_epoch_{epoch}.pth'
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = epoch
            print(f'Resuming training from epoch {start_epoch}')
            break

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            images = images.reshape(-1, 1, 28, 28)

            # one-hot encode the labels
            labels = nn.functional.one_hot(labels, num_classes=10).to(
                torch.float32).squeeze()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(
            f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}'
        )

        # Save the model state
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'classifier_states/model_epoch_{epoch+1}.pth')

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            images = images.reshape(-1, 1, 28, 28)
            labels = labels.reshape(-1)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')

    return model, 100 * correct / total


def show_classification(file_name,
                        img,
                        cmap='gray',
                        vmin=None,
                        vmax=None,
                        save=False,
                        savename='',
                        labels=None):
    """Display an image or a grid of images with optional labels and save if specified.

    Parameters
    ----------
    file_name : str
        Title for the plot.
    img : numpy array
        Image or grid of images to display.
    cmap : str, optional
        Colormap to use for `plt.imshow` (default is 'gray').
    vmin : float, optional
        Minimum data value that corresponds to colormap 'under' value.
    vmax : float, optional
        Maximum data value that corresponds to colormap 'over' value.
    save : bool, optional
        If True, saves the plot to `savename` (default is False).
    savename : str, optional
        Filename to save the plot (default is '').
    labels : list, optional
        Optional 1D list of labels for a grid of images.
    """
    plt.figure(figsize=(8, 8))
    plt.title(file_name)
    plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)

    # Overlay labels if provided
    if labels is not None:
        num_images = len(labels)
        grid_size = int(num_images**0.5)

        # Ensure labels length matches the grid size
        if grid_size * grid_size != num_images:
            raise ValueError('The length of labels must be a perfect square.')

        # Calculate sub-image dimensions
        img_height, img_width = img.shape[0] // grid_size, img.shape[
            1] // grid_size

        # Place labels
        for idx, label in enumerate(labels):
            row, col = divmod(idx, grid_size)
            plt.text(
                col * img_width + img_width,  # x-coordinate
                row * img_height,  # y-coordinate
                label,  # Label text
                color='red',
                fontsize=10,
                ha='center',
                va='center',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    if save:
        plt.savefig(savename)
        plt.close()
    else:
        plt.show()
