from base import Network, train, test
import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn
import matplotlib.pyplot as plt
from PIL import Image
import sys

# A transformation for greek letters
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale( x )
        x = torchvision.transforms.functional.affine( x, 0, (0,0), 36/128, 0 )
        x = torchvision.transforms.functional.center_crop( x, (28, 28) )
        return torchvision.transforms.functional.invert( x )
    
# A transformation for bigger greek letters
class ExtensiveGreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale( x )
        x = torchvision.transforms.functional.affine( x, 0, (0,0), 0.8, 0 )
        x = torchvision.transforms.functional.center_crop( x, (28, 28) )
        return torchvision.transforms.functional.invert( x )

# Transfer learning the last classification layer, while keeping the other layers frozen and trains the model to predict 3 classes
def transfer_learning(network, greek_dataset, device, num_classes = 3):
    # Freeze all classification layers
    for param in network.parameters():
        param.requires_grad = False
    # Change the last classification layer
    features = network.fc2.in_features
    network.fc2 = nn.Linear(features, num_classes)

    optimizer = torch.optim.Adam(network.parameters(), lr = 0.001)
    network.to(device)
    test_losses = []
    # Train for 11 epochs
    for i in range(11):
        train_loss, train_accuracy = train(network, device, greek_dataset, optimizer, 10)
        test_loss, test_accuracy = test(network, device, greek_dataset)
        test_losses.append(test_loss)
        print(f'Loss: {test_loss}, Accuracy: {test_accuracy}')

    # Plot the test losses
    plt.figure(figsize=(10,10))
    plt.plot(test_losses)
    plt.show()

# Fine-tunes on a bigger size problem by retraining the last two classification layer
def transfer_learning_bigger(network, greek_dataset, device, num_classes = 3):
    # Freeze all layers
    for param in network.parameters():
        param.requires_grad = False
    features1 = network.fc1.in_features
    #replace two last layers
    network.fc1 = nn.Linear(features1, 50)
    network.fc2 = nn.Linear(50, num_classes)

    optimizer = torch.optim.Adam(network.parameters(), lr = 0.001)
    network.to(device)
    test_losses = []

    # Train for 20 epochs
    for i in range(20):
        train_loss, train_accuracy = train(network, device, greek_dataset, optimizer, 10)
        test_loss, test_accuracy = test(network, device, greek_dataset)
        test_losses.append(test_loss)
        print(f'Loss: {test_loss}, Accuracy: {test_accuracy}')

    plt.figure(figsize=(10,10))
    plt.plot(test_losses)
    plt.show()

classes = ['alpha', 'beta', 'delta', 'epsilon', 'fi', 'gamma', 'heta', 'iota', 'kappa', 'ksi', 'lambda', 'mi', 'ni', 'omega', 'omikron', 'pi']


# Predicts the class of an bunch of images
def predict(network, device, test_loader):
    network.to(device)
    network.eval()
    preds = []
    images_to_plot = []
    # for each image, hets predictions and sves them in a list
    for image, _ in test_loader:
        image = image.to(device)
        output = network(image)
        pred = output.argmax(dim=1, keepdim=True)
        preds.append(pred.item())
        images_to_plot.extend(image.cpu())
    return preds, images_to_plot

# Predicts the class of a single image, does the whole classification path and return the label
def predict_image(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # opens the image
    image = Image.open(image_path).convert('L')
    # applies necessary transformations
    transform = ExtensiveGreekTransform()
    image_tensor = transforms.functional.to_tensor(transform(image)).unsqueeze(0)
    # loads the model
    network = Network()
    features1 = network.fc1.in_features
    network.fc1 = nn.Linear(features1, 50)
    network.fc2 = nn.Linear(50, 16)
    network.load_state_dict(torch.load('greek_model.pth', map_location=device))
    network.to(device)
    # sends the image to the device
    image_tensor = image_tensor.to(device)
    network.eval()
    # makes the prediction
    output = network(image_tensor)
    pred = output.argmax(dim=1, keepdim=True)
    return classes[pred.item()]

# Plots the results of predictions
def plot_results(preds, images_to_plot, classes):
    
    plt.figure(figsize=(15, 10))
    for i, (image, pred) in enumerate(zip(images_to_plot, preds)):
        plt.subplot(3, len(preds)//3+1, i+1)
        plt.imshow(image.squeeze().cpu().numpy(), cmap='gray')
        plt.title(f'Predicted: {classes[pred]}')
        plt.axis('off')
    plt.show()

# learns on a bigger dataset with more classes
def learn_bigger_network():
    # transformation for greek letters
    transform = torchvision.transforms.Compose( [torchvision.transforms.ToTensor(),
                                                ExtensiveGreekTransform(),
                                                torchvision.transforms.Normalize(
                                                    (0.1307,), (0.3081,) ) ] )
    # loads the dataset
    greek_dataset = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder( "greek_more", transform = transform), batch_size = 10, shuffle = True )

    #loads the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = Network().to(device)
    network.load_state_dict(torch.load('mnist_model.pth', map_location=device))
    # fine-tunes the model
    transfer_learning_bigger(network, greek_dataset, device, 16)

    # loads the test dataset
    greek_test = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder( "greek_more_test", transform = transform), batch_size = 1, shuffle = True )
    test_loss, test_accuracy = test(network, device, greek_test)

    print(f'Loss on drawn letters: {test_loss}, Accuracy on drawn letters: {test_accuracy}')
    # clasification classes
    classes = ['alpha', 'beta', 'delta', 'epsilon', 'fi', 'gamma', 'heta', 'iota', 'kappa', 'ksi', 'lambda', 'mi', 'ni', 'omega', 'omikron', 'pi']
    #classes = ['alpha', 'beta', 'delta', 'epsilon', 'fi', 'gamma', 'heta', 'iota', 'kappa', 'lambda', 'mu', 'nu', 'omicron', 'pi', 'rho', 'sigma', 'tau', 'theta', 'upsilon', 'phi', 'chi', 'psi', 'omega', 'zeta']
    plot_results(*predict(network, device, greek_test), classes)
    # saves the model
    torch.save(network.state_dict(), 'greek_model.pth')

def main(argv):
    if len(argv) > 1 and argv[1] == 'bigger':
        #trains the 2 layers network on 16 classes
        learn_bigger_network()
    else:
        # trains the network on 3 classes
        # transformation for greek letters
        transform = torchvision.transforms.Compose( [torchvision.transforms.ToTensor(),
                                                    GreekTransform(),
                                                    torchvision.transforms.Normalize(
                                                        (0.1307,), (0.3081,) ) ] )
        greek_dataset = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder( "greek_train", transform = transform), batch_size = 3, shuffle = True )

        # loads the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        network = Network().to(device)
        network.load_state_dict(torch.load('mnist_model.pth', map_location=device))
        # fine-tunes the model on 3 classes
        transfer_learning(network, greek_dataset, device, 3)

        #loads the hand written dataset
        greek_test = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder( "greek_test", transform = transform), batch_size = 1, shuffle = True )
        #evaluates the model on the hand written dataset
        test_loss, test_accuracy = test(network, device, greek_test)

        print(f'Loss on drawn letters: {test_loss}, Accuracy on drawn letters: {test_accuracy}')
        # clasification classes
        classes = ['alpha', 'beta', 'gamma']
        #plot the results
        plot_results(*predict(network, device, greek_test), classes)

    

if __name__ == '__main__':
    main(sys.argv)