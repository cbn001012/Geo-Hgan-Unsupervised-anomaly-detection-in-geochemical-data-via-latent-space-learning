import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Z_Encoder(nn.Module):
    def __init__(self, img_shape, latent_dim):
        super(Z_Encoder, self).__init__()

        # Encoder layers to map input image to latent space
        self.encode = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 1024),  # Fully connected layer 1
            nn.LeakyReLU(0.2, inplace=True),  # Activation function 1

            nn.Linear(1024, 512),  # Fully connected layer 2
            nn.BatchNorm1d(512),  # Batch normalization layer 2
            nn.LeakyReLU(0.2, inplace=True),  # Activation function 2

            nn.Linear(512, 256),  # Fully connected layer 3
            nn.BatchNorm1d(256),  # Batch normalization layer 3
            nn.LeakyReLU(0.2, inplace=True),  # Activation function 3

            nn.Linear(256, 128),  # Fully connected layer 4
            nn.BatchNorm1d(128),  # Batch normalization layer 4
            nn.LeakyReLU(0.2, inplace=True),  # Activation function 4

            nn.Linear(128, latent_dim),  # Fully connected layer 5 (output layer)
            nn.LeakyReLU(0.2, inplace=True)  # Activation function 5
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)  # Flatten the input image
        latent_space = self.encode(img_flat)  # Encode the flattened image to latent space
        return latent_space


class Z_Decoder(nn.Module):
    def __init__(self, img_shape, latent_dim):
        super(Z_Decoder, self).__init__()
        self.img_shape = img_shape
        self.latent_dim = latent_dim

        # Decoder layers to map latent space to reconstructed image
        self.decode = nn.Sequential(
            nn.Linear(self.latent_dim, 128),  # Fully connected layer 1
            nn.LeakyReLU(0.2, inplace=True),  # Activation function 1

            nn.Linear(128, 256),  # Fully connected layer 2
            nn.BatchNorm1d(256),  # Batch normalization layer 2
            nn.LeakyReLU(0.2, inplace=True),  # Activation function 2

            nn.Linear(256, 512),  # Fully connected layer 3
            nn.BatchNorm1d(512),  # Batch normalization layer 3
            nn.LeakyReLU(0.2, inplace=True),  # Activation function 3

            nn.Linear(512, 1024),  # Fully connected layer 4
            nn.BatchNorm1d(1024),  # Batch normalization layer 4
            nn.LeakyReLU(0.2, inplace=True),  # Activation function 4

            nn.Linear(1024, int(np.prod(self.img_shape))),  # Fully connected layer 5 (output layer)
            nn.Sigmoid()  # Sigmoid activation function to output values between 0 and 1
        )

    def forward(self, latent_space):
        x = self.decode(latent_space)  # Decode the latent space to the reconstructed image
        x = x.view(-1, self.img_shape[0], self.img_shape[1], self.img_shape[2])  # Reshape the image
        return x


class AE(nn.Module):
    def __init__(self, img_shape, latent_dim):
        super(AE, self).__init__()
        self.encoder = Z_Encoder(img_shape, latent_dim)  # Instantiate the encoder
        self.decoder = Z_Decoder(img_shape, latent_dim)  # Instantiate the decoder

    def forward(self, img):
        encoder_z = self.encoder(img)  # Encode the input image to latent space
        rs_img = self.decoder(encoder_z)  # Reconstruct the image from latent space
        return encoder_z, rs_img