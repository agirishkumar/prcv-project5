# Authors: Girish Kumar Adari, Alexander Seljuk
# Code for Task 1 A,B,C,D: Get the MNIST digit data set, Build a network model, Train the model, Save the network to a file

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from torchviz import make_dot

def visualize_mnist_test(test_loader):
    """
    Visualizes a batch of MNIST test images.

    Args:
        test_loader (torch.utils.data.DataLoader): The data loader for the MNIST test dataset.

    Returns:
        None

    This function takes a data loader for the MNIST test dataset and visualizes a batch of test images. It retrieves the images and labels from the data loader, moves the images back to the CPU for visualization, and then plots the images along with their corresponding labels. The function creates a figure with 6 subplots and displays the plot.

    Note:
        - This function assumes that the MNIST test dataset is already loaded and the data loader is provided as input.
        - The images are plotted in grayscale.
        - The function does not return any values.

    Example:
        visualize_mnist_test(test_loader)
    """
    dataiter = iter(test_loader)
    images, labels = next(dataiter)  
    images = images.to('cpu')  # Move images back to CPU for visualization

    # Plotting
    fig, axes = plt.subplots(1, 6, figsize=(12, 2))
    for i, ax in enumerate(axes):
        ax.imshow(images[i].numpy().squeeze(), cmap='gray')
        ax.set_title(f'Label: {labels[i]}')
        ax.axis('off')
    plt.show()



class Network(nn.Module):
    def __init__(self, in_channels=1, num_filters1=10, num_filters2=20, fc1_size=50, fc2_size=10):
        """
        Initializes the Network class with the given parameters.

        Args:
            in_channels (int): Number of input channels (default is 1)
            num_filters1 (int): Number of filters for the first convolutional layer (default is 10)
            num_filters2 (int): Number of filters for the second convolutional layer (default is 20)
            fc1_size (int): Size of the first fully connected layer (default is 50)
            fc2_size (int): Size of the second fully connected layer (default is 10)
        """
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, num_filters1, kernel_size=5)
        self.conv2 = nn.Conv2d(num_filters1, num_filters2, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(320, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)

    def forward(self, x):
        """
        Forward pass through the neural network.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, num_classes).
        """
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    
def train(model, device, train_loader, optimizer, epoch):
    """
    Trains the model using the provided data and optimizer for one epoch.

    Args:
        model: The neural network model to be trained.
        device: The device where the model and data will be processed.
        train_loader: The data loader containing the training data.
        optimizer: The optimizer used to update the model parameters.
        epoch: The current epoch number.

    Returns:
        batch_losses: A list of losses for each batch during the epoch.
        batch_accuracies: A list of accuracies for each batch during the epoch.
        cumulative_samples: A list of cumulative samples processed after each batch.
    """
    model.train()
    batch_losses = []
    batch_accuracies = []
    cumulative_samples = []
    total_samples = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        total_samples += target.size(0)
        batch_losses.append(loss.item())
        batch_accuracies.append(100. * correct / target.size(0))
        cumulative_samples.append(total_samples)
    return batch_losses, batch_accuracies, cumulative_samples

def test(model, device, test_loader):
    """
    Function to test a model on a given test dataset.

    Parameters:
    - model: the neural network model to be tested
    - device: the device on which the model is evaluated
    - test_loader: the data loader for the test dataset

    Returns:
    - batch_losses: a list of losses calculated for each batch
    - batch_accuracies: a list of accuracies calculated for each batch
    - cumulative_samples: a list of cumulative samples processed
    """
    model.eval()
    batch_losses = []
    batch_accuracies = []
    cumulative_samples = []
    total_samples = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target, reduction='sum')
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            total_samples += target.size(0)
            batch_losses.append(loss.item())
            batch_accuracies.append(100. * correct / target.size(0))
            cumulative_samples.append(total_samples)
    return batch_losses, batch_accuracies, cumulative_samples



def main():
    """
    Main function that runs the training and testing loop for a neural network model on the MNIST dataset.
    
    Parameters:
    None
    
    Returns:
    None
    """
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # MNIST Training dataset and DataLoader
    train_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    # MNIST Test dataset and DataLoader
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transform,
    )
    test_loader = DataLoader(test_data, batch_size=6, shuffle=False)

    # Visualize the first six test digits
    visualize_mnist_test(test_loader)

    model = Network().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    x = torch.randn(1, 1, 28, 28).to(device)  # Ensure the input tensor is on the correct device
    out = model(x)  # Forward pass to get the model output
    dot = make_dot(out, params=dict(list(model.named_parameters()) + [('x', x)]))  # Create the visualization
    dot.render("Network_architecture", format="png")

    epochs = 5
    # Initialize lists to store batch losses and accuracies
    train_batch_losses, train_batch_accuracies, train_cumulative_samples = [], [], []
    test_batch_losses, test_batch_accuracies, test_cumulative_samples = [], [], []

    for epoch in range(1, epochs + 1):
        batch_losses, batch_accuracies, cumulative_samples = train(model, device, train_loader, optimizer, epoch)
        train_batch_losses.extend(batch_losses)
        train_batch_accuracies.extend(batch_accuracies)
        train_cumulative_samples.extend(cumulative_samples)

        batch_losses, batch_accuracies, cumulative_samples = test(model, device, test_loader)
        test_batch_losses.extend(batch_losses)
        test_batch_accuracies.extend(batch_accuracies)
        test_cumulative_samples.extend(cumulative_samples)

    # Plotting the losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_cumulative_samples, train_batch_losses, label='Train Batch Loss', color='green')
    plt.plot(test_cumulative_samples, test_batch_losses, label='Test Batch Loss', color='red')
    plt.title('Negative Log Likelihood Loss vs Number of Samples')
    plt.xlabel('Number of Samples Seen')
    plt.ylabel('Negative Log Likelihood Loss')
    plt.legend()
    plt.show()

    # Plotting the accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(train_cumulative_samples, train_batch_accuracies, label='Train Batch Accuracy', color='blue')
    plt.plot(test_cumulative_samples, test_batch_accuracies, label='Test Batch Accuracy', color='orange')
    plt.title('Accuracy vs Number of Samples')
    plt.xlabel('Number of Samples Seen')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.show()
    
    # Save the model
    torch.save(model.state_dict(), 'mnist_model.pth')
    print('Model saved to mnist_model.pth')
 

if __name__ == "__main__":
    main()