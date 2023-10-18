import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tools import loadTifImage
from nets.sample_GAN import WGAN_Generator, WGAN_Discriminator
from nets.autoencoder import Z_Encoder
from skimage import io
import sys

# Set the current working directory (modify the directory according to your own situation)
sys.path.append('/mnt/3.6T-DATA/CBN/code/latent_space_learning/')

# Specify the GPU used for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
Set the relevant hyperparameters:
channels: Channel of the samples.
n_critic: The number of times the critic (discriminator) is updated per generator update in the training process of a GAN.
img_size: Size of the samples.
latent_dim: Dimension of the latent space features.
batch_size: Batch size for training. 
n_epochs: Number of epochs for training. 
b1: The beta1 parameter used in the optimizer. 
b2: The beta2 parameter used in the optimizer.
'''
channels = 39
n_critic = 5
img_size = 64
latent_dim = 100
batch_size = 64
n_epochs = 100
b1 = 0.5
b2 = 0.999

img_shape = (channels, img_size, img_size)

# Load the training dataset (only normal (negative) data is used for training)
train_dataset = loadTifImage.DatasetFolder(root='/mnt/3.6T-DATA/CBN/DATA/dim39_zheng_anomaly/train/',
                                          transform=transforms.ToTensor()) # "root" should be modified to the path of your training dataset
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Load the encoder
encoder = Z_Encoder(img_shape=img_shape,latent_dim=latent_dim)
# Load the pre-trained weights of the encoder in the latent space learning module and continue training based on that
encoder.load_state_dict(torch.load("/mnt/3.6T-DATA/CBN/train_wgangp_latent/dim39_attention/encoder.pth"))

# Load the pre-trained weights of the sample adversarial generator and sample adversarial discriminator in the latent space learning module
generator = torch.load("/mnt/3.6T-DATA/CBN/train_wgangp_latent/dim39_attention/G_wgangp.pth")
discriminator = torch.load("/mnt/3.6T-DATA/CBN/train_wgangp_latent/dim39_attention/D_wgangp.pth")

# Utilize the pre-trained GAN to constrain the variation of feature extraction in the encoder
def train_encoder_izif(generator, discriminator, encoder, dataloader, device, kappa=1.0):

    generator.to(device).eval() # freeze weights
    discriminator.to(device).eval() # freeze weights
    encoder.to(device)

    criterion = nn.MSELoss() # reconstructed loss

    optimizer_E = torch.optim.Adam(encoder.parameters(), lr=0.0001, betas=(b1,b2)) # optimizer of the encoder

    # Create a folder to save the generated images. Please modify the path to your own directory.
    os.makedirs("/mnt/3.6T-DATA/CBN/train_wgangp_latent/dim39_attention/encoderizif_generated_images", exist_ok=True)

    padding_epoch = len(str(n_epochs))
    padding_i = len(str(len(dataloader)))
    batches_done = 0

    # Training loop
    for epoch in range(n_epochs):
        for i, (imgs,labels) in enumerate(dataloader):

            # Configure input
            real_imgs = imgs.to(device)

            # ----------------
            #  Train Encoder
            # ----------------

            optimizer_E.zero_grad()

            # Generate a batch of latent variables
            z = encoder(real_imgs)

            # Generate a batch of images
            fake_imgs = generator(z)

            # Real features
            real_features = discriminator.forward_features(real_imgs)
            # Fake features
            fake_features = discriminator.forward_features(fake_imgs)

            real_features = real_features / real_features.max()
            fake_features = fake_features / fake_features.max()

            # izif architecture
            loss_imgs = criterion(fake_imgs, real_imgs)
            loss_features = criterion(fake_features, real_features)
            e_loss = loss_imgs + kappa * loss_features

            e_loss.backward()
            optimizer_E.step()

            # Output training log every n_critic steps
            if i % n_critic == 0:
                print(f"[Epoch {epoch:{padding_epoch}}/{n_epochs}] "
                      f"[Batch {i:{padding_i}}/{len(dataloader)}] "
                      f"[E loss: {e_loss.item():3f}]")

                # Save generated images every 400 iterations
                if batches_done % 400 == 0:
                    fake_z = encoder(fake_imgs)
                    reconfiguration_imgs = generator(fake_z)
                    io.imsave(f"/mnt/3.6T-DATA/CBN/train_wgangp_latent/dim39_attention/"
                              f"encoderizif_generated_images/{batches_done:06}.tif",reconfiguration_imgs.data.cpu().numpy())
                batches_done += 5

    # save the model weights (the path should be modified to your own directory)
    torch.save(encoder, "/mnt/3.6T-DATA/CBN/train_wgangp_latent/dim39_attention/encoderizif.pth")

# begin train!
train_encoder_izif(generator, discriminator, encoder, train_dataloader, device)
