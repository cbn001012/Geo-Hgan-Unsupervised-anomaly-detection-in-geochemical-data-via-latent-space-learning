import torch.nn as nn


class latent_Generator(nn.Module):
    def __init__(self, z_dim):
        super(latent_Generator, self).__init__()
        self.z_dim = z_dim

        # Generator layers to map random noise to generated latent space features
        self.model = nn.Sequential(
            nn.Linear(self.z_dim, 512),  # Fully connected layer 1
            nn.BatchNorm1d(512),  # Batch normalization layer 1
            nn.LeakyReLU(0.2, inplace=True),  # Activation function 1

            nn.Linear(512, 1024),  # Fully connected layer 2
            nn.BatchNorm1d(1024),  # Batch normalization layer 2
            nn.LeakyReLU(0.2, inplace=True),  # Activation function 2

            nn.Linear(1024, 512),  # Fully connected layer 3
            nn.BatchNorm1d(512),  # Batch normalization layer 3
            nn.LeakyReLU(0.2, inplace=True),  # Activation function 3

            nn.Linear(512, z_dim)  # Fully connected layer 4 (output layer)
        )

    def forward(self, z):
        latent_z = self.model(z)
        return latent_z


class latent_Discriminator(nn.Module):
    def __init__(self, z_dim):
        super(latent_Discriminator, self).__init__()
        self.z_dim = z_dim

        # Discriminator layers to classify latent space features as real or fake
        self.features = nn.Sequential(
            nn.Linear(z_dim, 512),  # Fully connected layer 1
            nn.BatchNorm1d(512),  # Batch normalization layer 1
            nn.LeakyReLU(0.2, inplace=True),  # Activation function 1

            nn.Linear(512, 1024),  # Fully connected layer 2
            nn.BatchNorm1d(1024),  # Batch normalization layer 2
            nn.LeakyReLU(0.2, inplace=True),  # Activation function 2

            nn.Linear(1024, 512),  # Fully connected layer 3
            nn.BatchNorm1d(512),  # Batch normalization layer 3
            nn.LeakyReLU(0.2, inplace=True)  # Activation function 3
        )

        self.last_layer = nn.Sequential(
            nn.Linear(512, 1)  # Fully connected layer 4 (output layer)
        )

    def forward(self, z):
        features = self.features(z)  
        out_logit = self.last_layer(features)  
        return out_logit