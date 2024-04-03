import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data loading
transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST(root='./data2', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data2', train=False, download=True, transform=transform)

class CustomCNN(nn.Module):
    def __init__(self, num_filters1=10, num_filters2=20, dropout_rate=0.5, kernel_size1=5, kernel_size2=5, activation='relu'):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, num_filters1, kernel_size=kernel_size1)
        self.conv2 = nn.Conv2d(num_filters1, num_filters2, kernel_size=kernel_size2)
        self.conv2_drop = nn.Dropout2d(dropout_rate)

        # Ensure kernel_size is stored as tuple for later indexing
        self.conv1.kernel_size = (kernel_size1, kernel_size1)
        self.conv2.kernel_size = (kernel_size2, kernel_size2)

        # Dynamically calculate FC layer input size
        fc_input_size = self._calculate_fc_input_size()

        self.fc1 = nn.Linear(fc_input_size, 50)
        self.fc2 = nn.Linear(50, 10)
        
        # Activation function handling (similar to your previous approach)
        self.activation = getattr(F, activation, F.relu)  # Default to F.relu if not found

    def _calculate_fc_input_size(self):
        # Initial size (assuming square images)
        size = 28  # Initial image size is 28x28 pixels

        # Apply first convolution and pooling
        size = (size - (self.conv1.kernel_size[0] - 1) - 1) // 2 + 1  # Conv1 and pooling

        # Apply second convolution and pooling
        size = (size - (self.conv2.kernel_size[0] - 1) - 1) // 2 + 1  # Conv2 and pooling

        # Calculate total number of features
        return size * size * self.conv2.out_channels

    def forward(self, x):
        x = self.activation(F.max_pool2d(self.conv1(x), 2))
        x = self.activation(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, self._calculate_fc_input_size())
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)




def train(model, device, train_loader, optimizer, epoch):
    """
    Trains a given model using the provided training data and optimizer.

    Args:
        model (torch.nn.Module): The model to be trained.
        device (torch.device): The device on which the model and data should be loaded.
        train_loader (torch.utils.data.DataLoader): The data loader for the training dataset.
        optimizer (torch.optim.Optimizer): The optimizer used for updating the model's parameters.
        epoch (int): The current epoch number.

    Returns:
        None
    """
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # print(f"Output size: {output.size()}") 
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

def test(model, device, test_loader):
    """
    Compute the test loss and accuracy for a given model.

    Args:
        model (torch.nn.Module): The model to evaluate.
        device (torch.device): The device to run the evaluation on.
        test_loader (torch.utils.data.DataLoader): The data loader for the test dataset.

    Returns:
        Tuple[float, float]: A tuple containing the test loss and accuracy.
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy

experiment_results = []

def run_experiment(num_filters1, num_filters2, dropout_rate, lr, batch_size=64, optimizer_choice='SGD', kernel_size1=5, kernel_size2=5, activation='relu', num_epochs=5):
    # Adjust DataLoader batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomCNN(num_filters1=num_filters1, num_filters2=num_filters2, dropout_rate=dropout_rate, kernel_size1=kernel_size1, kernel_size2=kernel_size2, activation=activation).to(device)

    # Choose optimizer
    if optimizer_choice == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5)
    elif optimizer_choice == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)

    start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
    train_time = time.time() - start_time

    test_loss, accuracy = test(model, device, test_loader)

    experiment_results.append({
        "num_filters1": num_filters1,
        "num_filters2": num_filters2,
        "dropout_rate": dropout_rate,
        "learning_rate": lr,
        "batch_size": batch_size,
        "optimizer": optimizer_choice,
        "activation": activation,
        "accuracy": accuracy
    })

    print(f"Experiment: Filters1={num_filters1}, Filters2={num_filters2}, Dropout={dropout_rate}, LR={lr}, BatchSize={batch_size}, Optimizer={optimizer_choice}, Kernel1={kernel_size1}, Kernel2={kernel_size2}, Activation={activation}, Epochs={num_epochs} - Accuracy: {accuracy:.2f}%, Time: {train_time:.2f}s")


# Define ranges for each parameter
num_filters1_range = [10, 20]
num_filters2_range = [10, 20, 40]
dropout_rate_range = [0.25, 0.5, 0.75]
lr_range = [0.01, 0.05, 0.1]
batch_size_range = [64, 128]
optimizer_choices = ['SGD', 'Adam']
activation_functions = ['relu']

def run_experiments():
    for num_filters1 in num_filters1_range:
        for num_filters2 in num_filters2_range:
            for dropout_rate in dropout_rate_range:
                for lr in lr_range:
                    for batch_size in batch_size_range:
                        for optimizer_choice in optimizer_choices:
                            for activation in activation_functions:
                                # Ensure num_filters2 is greater or equal to num_filters1
                                if num_filters2 >= num_filters1:
                                    run_experiment(num_filters1=num_filters1, num_filters2=num_filters2, dropout_rate=dropout_rate, lr=lr, batch_size=batch_size, optimizer_choice=optimizer_choice, activation=activation)

                                    
df = pd.DataFrame(experiment_results)
df.to_csv('experiment_results.csv', index=False)

def plot_parameter_impact(df, parameter_name):
    grouped_data = df.groupby(parameter_name)['accuracy'].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    if np.issubdtype(grouped_data[parameter_name].dtype, np.number):
        plt.plot(grouped_data[parameter_name], grouped_data['accuracy'], marker='o', linestyle='-')
    else:
        plt.bar(grouped_data[parameter_name], grouped_data['accuracy'])
        plt.xticks(rotation=45, ha="right")
    plt.title(f'Impact of {parameter_name} on Accuracy')
    plt.xlabel(parameter_name)
    plt.ylabel('Average Accuracy (%)')
    plt.grid(True)
    plt.show()
                                
if __name__ == "__main__":
    if os.path.exists('experiment_results2.csv'):
        # Check if the file is not empty
        if os.path.getsize('experiment_results.csv') > 0:
            print("Experiment results file found. Reading...")
            df = pd.read_csv('experiment_results2.csv')
            # Check if DataFrame is not empty and has at least one column
            if df.empty or df.columns.empty:
                print("CSV file found but it is empty or malformed. Running experiments...")
                os.remove('experiment_results.csv')  # Remove the empty/malformed file
                run_experiments()
                df = pd.DataFrame(experiment_results)
                df.to_csv('experiment_results2.csv', index=False)
        else:
            print("CSV file found but it is empty. Running experiments...")
            run_experiments()
            df = pd.DataFrame(experiment_results)
            
    else:
        print("Experiment results file not found. Running experiments...")
        run_experiments()
        
    df['filter_combo'] = df['filter_combo'].apply(eval)
    grouped_data = df.groupby(['filter_combo', 'optimizer'])['accuracy'].mean().unstack()
    grouped_data.index = grouped_data.index.map(lambda x: f'({x[0]},{x[1]})')
    


    # Continue with your plotting functions if df is not empty
    print("Plotting results...")
    plot_parameter_impact(df, 'learning_rate')
    plot_parameter_impact(df, 'dropout_rate')
    plot_parameter_impact(df, 'batch_size')
    plot_parameter_impact(df, 'optimizer')
    plot_parameter_impact(df, 'num_filters1')
    plot_parameter_impact(df, 'num_filters2')
    fig, ax = plt.subplots(figsize=(12, 8))
    grouped_data.plot(kind='bar', ax=ax)
    plt.xlabel('(num_filters1, num_filters2)')
    plt.ylabel('Average Accuracy (%)')
    plt.title('Average Accuracy for Different Filter Combinations and Optimizers')
    plt.xticks(rotation=45)
    plt.legend(title='Optimizer')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()



 
    


