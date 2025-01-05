import torch
import torch.nn as nn


class FullyConvNetwork(nn.Module):

    def __init__(self):
        super(FullyConvNetwork, self).__init__()

        # Encoder (Convolutional Layers)
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # Input channels: 3, Output channels: 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        # Decoder (Deconvolutional Layers)
        self.upconv5 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)  # Output channels for RGB

        # Skip connections
        self.skip_conv4 = nn.Conv2d(512, 512, kernel_size=1)
        self.skip_conv3 = nn.Conv2d(256, 256, kernel_size=1)
        self.skip_conv2 = nn.Conv2d(128, 128, kernel_size=1)

    def forward(self, x):
        # Encoder forward pass
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)

        # Decoder forward pass with skip connections
        dec1 = self.upconv5(enc5)
        dec1 += self.skip_conv4(enc4)  # Skip connection from enc4
        dec2 = self.upconv4(dec1)
        dec2 += self.skip_conv3(enc3)  # Skip connection from enc3
        dec3 = self.upconv3(dec2)
        dec3 += self.skip_conv2(enc2)  # Skip connection from enc2
        dec4 = self.upconv2(dec3)

        # Final output
        output = self.final_conv(dec4)

        return output
