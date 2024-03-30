import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from base import Network
from analyze import load_model

# Assuming you've already defined the Network class and load_model function

def apply_filters_and_visualize(model, device, train_loader):
    # Get the first image from the MNIST training dataset
    images, _ = next(iter(train_loader))
    image = images[0].squeeze().cpu().numpy()  # Convert to numpy array and remove channel dimension

    # Get the weights of the first layer (conv1)
    with torch.no_grad():
        filters = model.conv1.weight.data.clone().cpu()

    # Normalize the filter values to 0-255
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min) * 255
    filters = filters.numpy().astype(np.uint8)
    
    # Plot original image
    plt.figure(figsize=(15, 10))
    plt.subplot(3, 4, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Apply each filter to the original image using OpenCV's filter2D function
    for i in range(10):
        filtered_img = cv2.filter2D(image, -1, filters[i, 0])
        plt.subplot(3, 4, i+2)
        plt.imshow(filtered_img, cmap='gray')
        plt.title(f'Filter {i+1}')
        plt.axis('off')

    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'mnist_model.pth'
    model = load_model(model_path, device)

    # Prepare the data loader
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)

    # Apply filters and visualize
    apply_filters_and_visualize(model, device, train_loader)

if __name__ == "__main__":
    main()
