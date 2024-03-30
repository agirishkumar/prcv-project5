import matplotlib.pyplot as plt
import numpy as np
# import hiddenlayer as hl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim

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
    images, labels = next(dataiter)  # Use next() function with the iterator
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


# Example of using the model
# model = Network()
# print(model)
# graph = hl.build_graph(model, torch.zeros([1, 1, 28, 28]))
# graph.save('Network.png', format='png')
    
def train(model, device, train_loader, optimizer, epoch):
    """
    Train the model using the given data and optimizer for the specified number of epochs.
    
    Args:
        model (torch.nn.Module): The model to be trained.
        device (torch.device): The device on which the model and data should be loaded.
        train_loader (torch.utils.data.DataLoader): The data loader for the training dataset.
        optimizer (torch.optim.Optimizer): The optimizer to be used for training.
        epoch (int): The current epoch number.
    
    Returns:
        tuple: A tuple containing the average loss and accuracy of the trained model on the training set.
    """
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    
    train_loss /= len(train_loader.dataset)
    accuracy = 100. * correct / len(train_loader.dataset)
    print(f'\nTrain set: Average loss: {train_loss:.4f}, Accuracy: {correct}/{len(train_loader.dataset)}'
          f' ({accuracy:.0f}%)\n')
    return train_loss, accuracy


def test(model, device, test_loader):
    """
    Evaluates the performance of a model on a test dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        device (torch.device): The device on which the model and data will be loaded.
        test_loader (torch.utils.data.DataLoader): The data loader for the test dataset.

    Returns:
        Tuple[float, float]: A tuple containing the average test loss and the accuracy of the model on the test dataset.
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}'
          f' ({accuracy:.0f}%)\n')
    return test_loss, accuracy


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

    epochs = 5
    # Initialize variables for storing metrics
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []
    
    # Training and testing loop
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train(model, device, train_loader, optimizer, epoch)
        test_loss, test_accuracy = test(model, device, test_loader)
        
        # Append metrics to their respective lists
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', color='green')
    plt.plot(test_losses, label='Test Loss', color='red')
    plt.title('Training vs Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Train Accuracy', color='green')
    plt.plot(test_accuracies, label='Test Accuracy', color='blue')
    plt.title('Training vs Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.show()

    # Save the model
    torch.save(model.state_dict(), 'mnist_model.pth')
    print('Model saved to mnist_model.pth')


    

if __name__ == "__main__":
    main()