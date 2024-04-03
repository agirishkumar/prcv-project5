from base import Network, train, test
import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn
import matplotlib.pyplot as plt

class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale( x )
        x = torchvision.transforms.functional.affine( x, 0, (0,0), 36/128, 0 )
        x = torchvision.transforms.functional.center_crop( x, (28, 28) )
        return torchvision.transforms.functional.invert( x )

def transfer_learning(network, greek_dataset, device):
    for param in network.parameters():
        param.requires_grad = False
    features = network.fc2.in_features
    network.fc2 = nn.Linear(features, 3)
    optimizer = torch.optim.Adam(network.parameters(), lr = 0.001)
    network.to(device)
    test_losses = []
    for i in range(13):
        train_loss, train_accuracy = train(network, device, greek_dataset, optimizer, 10)
        test_loss, test_accuracy = test(network, device, greek_dataset)
        test_losses.append(test_loss)
        print(f'Loss: {test_loss}, Accuracy: {test_accuracy}')

    plt.figure(figsize=(10,10))
    plt.plot(test_losses)
    plt.show()

def predict(network, device, test_loader):
    network.to(device)
    network.eval()
    preds = []
    images_to_plot = []
    for image, _ in test_loader:
        image = image.to(device)
        output = network(image)
        pred = output.argmax(dim=1, keepdim=True)
        preds.append(pred.item())
        images_to_plot.extend(image.cpu())
    return preds, images_to_plot

def plot_results(preds, images_to_plot, classes):
    
    plt.figure(figsize=(15, 10))
    for i, (image, pred) in enumerate(zip(images_to_plot, preds)):
        plt.subplot(3, 3, i+1)
        plt.imshow(image.squeeze().cpu().numpy(), cmap='gray')
        plt.title(f'Predicted: {classes[pred]}')
        plt.axis('off')
    plt.show()

def main():
    transform = torchvision.transforms.Compose( [torchvision.transforms.ToTensor(),
                                                GreekTransform(),
                                                torchvision.transforms.Normalize(
                                                    (0.1307,), (0.3081,) ) ] )
    greek_dataset = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder( "greek_train", transform = transform), batch_size = 2, shuffle = True )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = Network().to(device)
    network.load_state_dict(torch.load('mnist_model.pth', map_location=device))
    transfer_learning(network, greek_dataset, device)

    greek_test = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder( "greek_test", transform = transform), batch_size = 1, shuffle = True )
    test_loss, test_accuracy = test(network, device, greek_test)

    print(f'Loss on drawn letters: {test_loss}, Accuracy on drawn letters: {test_accuracy}')

    classes = ['alpha', 'beta', 'gamma']
    plot_results(*predict(network, device, greek_test), classes)

    

if __name__ == '__main__':
    main()