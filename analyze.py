import torch
from base import Network
import matplotlib.pyplot as plt
import math

def load_model(model_path, device):
    """
    Load a model from the given model_path onto the specified device.

    Args:
        model_path (str): The file path to the model.
        device: The device onto which the model will be loaded.

    Returns:
        model: The loaded model.
    """
    model = Network().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def print_model(model):
    """
    Print the given model.
    
    Parameters:
    model (any): The model to be printed.
    
    Returns:
    None
    """
    print(model)

def visualize_filters(layer_weights):
    """
    Visualizes the filters of a given layer in a grid.
    
    Args:
        layer_weights (torch.Tensor): The weights of the layer to visualize.
        
    Returns:
        None
    """
    with torch.no_grad():  # We don't need gradients for visualization
        # Normalize the weights for visualization
        min_val = layer_weights.min()
        max_val = layer_weights.max()
        filters = (layer_weights - min_val) / (max_val - min_val)
        
        num_filters = filters.shape[0]
        columns = 4
        rows = math.ceil(num_filters / columns)  # Calculate rows, rounding up
        fig = plt.figure(figsize=(10, 2 * rows))
        for i in range(num_filters):
            ax = fig.add_subplot(rows, columns, i + 1)
            ax.imshow(filters[i, 0].cpu().numpy(), cmap='Greens')
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()

def main():
    """
    The main function that runs the program.

    This function does the following:
    1. Checks if a CUDA-enabled GPU is available and sets the device accordingly.
    2. Specifies the path to the model file.
    3. Loads and prints the model.
    4. Retrieves the weights of the first layer 'conv1' from the model.
    5. Prints the shape of the 'conv1' weights.
    6. Visualizes the filters of the first layer.

    Parameters:
    None

    Returns:
    None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'mnist_model.pth'

    # Load and print the model
    model = load_model(model_path, device)
    print_model(model)

    # Get the weights of the first layer 'conv1'
    first_layer_weights = model.conv1.weight.data
    print("Shape of conv1 weights:", first_layer_weights.shape)

    # Visualize the filters of the first layer
    visualize_filters(first_layer_weights)

if __name__ == "__main__":
    main()
