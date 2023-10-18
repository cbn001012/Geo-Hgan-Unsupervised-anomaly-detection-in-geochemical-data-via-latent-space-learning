import torch.nn as nn
import torch.nn.functional as F
import torch

# CBAM Attention Mechanism
class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

# Sample Adversarial Generator
class WGAN_Generator(nn.Module):
    def __init__(self, img_size, latent_dim, channels):
        super().__init__()

        self.init_size = img_size //  4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 64 * self.init_size ** 2))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 48, 3, stride=1, padding=1),
            nn.BatchNorm2d(48, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(48, channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 64, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# Sample Adversarial Discriminator
class WGAN_Discriminator(nn.Module):
    def __init__(self,channels,img_size):
        super().__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.utils.spectral_norm(nn.Conv2d(in_filters, out_filters, 3, 2, 1)),
                     nn.LeakyReLU(0.2, inplace=True)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(channels, 48, bn=False),
            *discriminator_block(48, 64, bn=False),
            *discriminator_block(64, 96, bn=False),
            *discriminator_block(96, 128, bn=False)
        )

        # The height and width of downsampled image
        ds_size = img_size // (2**4)
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1))
        self.cbam = CBAMLayer(channel=128)
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, img):
        features = self.forward_features(img)
        validity = self.adv_layer(features)
        return validity

    def forward_features(self, img):
        features = self.model(img)
        features_1 = self.cbam(features)
        features = features + features_1
        features = self.leakyrelu(features)
        features = features.view(features.shape[0], -1)
        return features

class Encoder(nn.Module):
    def  __init__(self, channels, img_size, latent_dim):
        super().__init__()

        def encoder_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), 
                     nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *encoder_block(channels, 48, bn=False),
            *encoder_block(48, 64),
            *encoder_block(64, 96),
            *encoder_block(96, 128)
        )

        # The height and width of downsampled image
        ds_size = img_size // (2 ** 4)
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2,
                                                 latent_dim),
                                       nn.Tanh())
        self.cbam = CBAMLayer(channel=128)
        self.leakyrelu = nn.LeakyReLU(0.2,inplace=True)

    def forward(self, img):
        features = self.model(img)
        features_1 = self.cbam(features)
        features = features + features_1
        features = self.leakyrelu(features)
        features = features.view(features.shape[0], -1)
        validity = self.adv_layer(features)
        return validity



