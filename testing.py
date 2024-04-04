# Authors: Girish Kumar Adari, Alexander Seljuk
# Code for Task 1 E,F: Read the network and run it on the test set, Test the network on new inputs.

import matplotlib.pyplot as plt
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from base import Network 
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF

def load_data(batch_size=10):
    """
    Loads the MNIST test data and returns a DataLoader object for batching the data.

    Parameters:
        batch_size (int, optional): The number of samples per batch. Defaults to 10.

    Returns:
        torch.utils.data.DataLoader: The DataLoader object for loading the test data in batches.
    """
    # Define the same transformation as used during training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load the test set
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transform
    )
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return test_loader

def predict(model, device, test_loader):
    """
    Set the model to evaluation mode, make predictions, print predicted and true labels, 
    print output values for each example, plot the first 9 images with their predicted and true labels.
    
    Parameters:
    - model: the neural network model
    - device: the device to run the model on
    - test_loader: the data loader for the test dataset
    
    Returns:
    None
    """
    model.eval()  # Set the model to evaluation mode
    test_iter = iter(test_loader)
    images, labels = next(test_iter)
    images, labels = images.to(device), labels.to(device)

    # Make predictions
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    # Print the predicted labels
    print('Predicted labels:', predicted.cpu().numpy())
    print('True labels:', labels.cpu().numpy())

    # Print the 10 output values for each example
    for i in range(len(labels)):
      print(f"Image {i}:")
      print("Output values: ", ['{:.2f}'.format(o) for o in outputs[i].detach().cpu().numpy()])
      print("Predicted label: ", predicted[i].item())
      print("True label: ", labels[i].item())
      print()

    # Plot the first 9 images
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for i, ax in enumerate(axes.ravel()):
        if i >= 9:
            break
        ax.imshow(images[i].cpu().numpy().squeeze(), cmap='gray')
        ax.set_title(f'Pred: {predicted[i].item()} / True: {labels[i].item()}')
        ax.axis('off')
    plt.show()

def preprocess_image(image_path):
    """
    Preprocesses an image by opening, converting to grayscale, inverting colors, converting to PyTorch tensor, normalizing, and adding a batch dimension.

    Args:
        image_path (str): The file path to the image.

    Returns:
        torch.Tensor: The preprocessed image tensor.
    """
    # Open the image file
    img = Image.open(image_path).convert('L')  # Convert to grayscale

    # Invert image colors to match MNIST
    img = ImageOps.invert(img)

    # Convert to PyTorch tensor
    img_tensor = TF.to_tensor(img)

    # Normalize the image to match MNIST dataset
    img_tensor = TF.normalize(img_tensor, [0.1307],[0.3081])

    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

def classify_digit(model, device, image_tensor):
    """
    Classifies a digit using a pre-trained model.

    Args:
        model (torch.nn.Module): The pre-trained model used for classification.
        device (torch.device): The device on which the model and tensor will be moved.
        image_tensor (torch.Tensor): The input image tensor.

    Returns:
        int: The predicted digit class.

    Note:
        - The model is set to evaluation mode before prediction.
        - The image tensor is moved to the specified device.
        - The output of the model is obtained by passing the image tensor.
        - The predicted digit class is returned as an integer.
    """
    # Set model to evaluation mode and move tensor to device
    model.eval()
    image_tensor = image_tensor.to(device)

    # Predict the digit
    output = model(image_tensor)
    pred = output.argmax(dim=1, keepdim=True)
    return pred.item()

def main():
    """
    The main function that runs the program.

    This function initializes and loads the model on the appropriate device (GPU if available, CPU otherwise). It then loads the pre-trained model weights from the 'mnist_model.pth' file. The function also loads the test data using the 'load_data' function with a batch size of 10. Finally, it calls the 'predict' function to make predictions using the loaded model and test data.

    Parameters:
    None

    Returns:
    None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize and load your model here
    model = Network().to(device)
    model.load_state_dict(torch.load('mnist_model.pth', map_location=device))

    test_loader = load_data(10)  # Load the data with a batch size of 10
    predict(model, device, test_loader)

    for i in range(10):
        image_path = f'{i}.JPG'  
        image_tensor = preprocess_image(image_path)
        prediction = classify_digit(model, device, image_tensor)
        print(f'Handwritten digit: {i}, Predicted digit: {prediction}')

if __name__ == "__main__":
    main()
