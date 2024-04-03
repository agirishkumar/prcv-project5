import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import save_image


# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self, nc=1, ndf=64):
        """
        Initializes the Discriminator class with default values for the number of input channels (nc) and number of discriminator filters (ndf).
        
        Parameters:
            nc (int): Number of input channels. Default is 1.
            ndf (int): Number of discriminator filters. Default is 64.
        """
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, 1, 7, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        """
        Perform forward pass with the given input.

        Args:
            input: The input tensor to be passed through the network.

        Returns:
            Tensor: The output tensor after passing through the network.
        """
        return self.main(input).view(-1, 1).squeeze(1)

class Generator(nn.Module):
    def __init__(self, nz, nc=1, ngf=64):
        """
        Initialize the Generator with the specified parameters.

        Parameters:
            nz (int): size of the input noise vector
            nc (int): number of channels in the output image (default is 1)
            ngf (int): size of the feature maps in the generator (default is 64)
        """
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 2, 7, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        """
        A description of the entire function, its parameters, and its return types.
        """
        return self.main(input)

# Hyperparameters
nz = 100  # Size of generator input 
lr = 0.0002
beta1 = 0.5
batch_size = 128
epochs = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the generator and discriminator models
netG = Generator(nz).to(device)
netD = Discriminator().to(device)

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Loss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Load data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = datasets.FashionMNIST(root='./data3', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training Loop
print("Starting Training Loop...")
for epoch in range(epochs):
    for i, data in enumerate(dataloader, 0):
        # Update discriminator: maximize log(D(x)) + log(1 - D(G(z)))
        netD.zero_grad()
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), 1, dtype=torch.float, device=device)
        output = netD(real_cpu)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(0)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        # Update generator: maximize log(D(G(z)))
        netG.zero_grad()
        label.fill_(1)
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

    # Check how the generator is doing by saving G's output on fixed_noise
    if (epoch % 1 == 0) or ((epoch == epochs-1) and (i == len(dataloader)-1)):
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        save_image(fake, 'output_epoch_%03d.png' % (epoch + 1), normalize=True)

