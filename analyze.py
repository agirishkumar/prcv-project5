import torch
from base import Network
import matplotlib.pyplot as plt
import math

def load_model(model_path, device):
    model = Network().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def print_model(model):
    print(model)

def visualize_filters(layer_weights):
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
            ax.imshow(filters[i, 0].cpu().numpy(), cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()

def main():
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
