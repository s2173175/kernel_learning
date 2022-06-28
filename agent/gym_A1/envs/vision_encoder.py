import sys
import os
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

LATENT_DIM = 10
CAPACITY = 64

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        c = CAPACITY
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=c, kernel_size=4, stride=2, padding=1) # out: c x 14 x 14
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=4, stride=2, padding=1) # out: c x 7 x 7
        self.fc = nn.Linear(in_features=c*2*7*7, out_features=LATENT_DIM)
            
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        x = self.fc(x)
        x = torch.clip(x, -1, 1)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        c = CAPACITY
        self.fc = nn.Linear(in_features=LATENT_DIM, out_features=c*2*7*7)
        self.conv2 = nn.ConvTranspose2d(in_channels=c*2, out_channels=c, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=c, out_channels=1, kernel_size=4, stride=2, padding=1)
            
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), CAPACITY*2, 7, 7) # unflatten batch of feature vectors to a batch of multi-channel feature maps
        x = F.relu(self.conv2(x))
        x = torch.tanh(self.conv1(x)) # last layer before output is tanh, since the images are normalized and 0-centered
        return x
    
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, x):
        latent = self.encoder(x)
        x_recon = self.decoder(latent)
        return x_recon


class A1_vision_encoder():
    def __init__(self, model_filename=None) -> None:
        assert model_filename, "Please provide a model_filename of the autoencoder."
        device = torch.device("cpu")
        self.autoencoder = Autoencoder()
        self.autoencoder = self.autoencoder.to(device)
        self.autoencoder.load_state_dict(
            torch.load(currentdir+f"/trained_autoencoder/{model_filename}", 
                       map_location=device
            )
        )

        self.autoencoder.double()
        self.autoencoder.eval()

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.0,), (1.0,)),
        ])

    def __call__(self, x):
        x = self.img_transform(x)
        x = torch.stack([x]).double()
        return self.autoencoder.encoder(x).cpu().detach().numpy().squeeze()
