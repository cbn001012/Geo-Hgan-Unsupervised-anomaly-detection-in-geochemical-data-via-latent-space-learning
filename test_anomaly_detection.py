import os
import sys
from genericpath import exists
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from skimage import io
from tools import loadTifImage
from nets.sample_GAN import WGAN_Generator,WGAN_Discriminator
from nets.autoencoder import Z_Encoder

# Set the current working directory (modify the directory according to your own situation)
sys.path.append('/mnt/3.6T-DATA/CBN/code/latent_space_learning/')

# Specify the GPU used for training
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Load the test dataset (modify the directory according to your own path of the test dataset)
test_dataset = loadTifImage.DatasetFolder(root='/mnt/3.6T-DATA/CBN/DATA/dim39_zheng_anomaly/test/',
                                          transform=transforms.ToTensor())
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

# Load the pre-trained parameters of the sample adversarial generator, sample adversarial discriminator, and encoder
generator = torch.load("/mnt/3.6T-DATA/CBN/train_wgangp_latent/dim39_attention/G_wgangp.pth")
discriminator = torch.load("/mnt/3.6T-DATA/CBN/train_wgangp_latent/dim39_attention/D_wgangp.pth")
encoder = Z_Encoder(img_shape=img_shape,latent_dim=latent_dim)
encoder.load_state_dict(torch.load("/mnt/3.6T-DATA/CBN/train_wgangp_latent/dim39_attention/encoderizif.pth")) # GAN-guided pretrained weights of the encoder

# test the performance of the anomaly detection on the test dataset
def test_anomaly_detection(generator, discriminator, encoder, dataloader, device, kappa=1.0):

    # freeze pre-trained weights of all models
    generator.to(device).eval()
    discriminator.to(device).eval()
    encoder.to(device).eval()

    criterion = nn.MSELoss() # reconstructed loss

    # Create a CSV file to save the anomaly scores
    with torch.no_grad():
        
        with open("/mnt/3.6T-DATA/CBN/Data_paper/dim39-attention-res.csv", "w") as f:
            f.write("label,img_distance,anomaly_score,z_distance\n")

        for i, (img, label) in enumerate(dataloader):
            real_img = img.to(device)

            real_z = encoder(real_img)
            fake_img = generator(real_z)
            fake_z = encoder(fake_img)

            real_feature = discriminator.forward_features(real_img)
            fake_feature = discriminator.forward_features(fake_img)

            # Scores for anomaly detection
            img_distance = criterion(fake_img, real_img)
            loss_feature = criterion(fake_feature, real_feature)
            anomaly_score = img_distance + kappa * loss_feature

            z_distance = criterion(fake_z, real_z)

            with open("/mnt/3.6T-DATA/CBN/Data_paper/dim39-attention-res.csv", "a") as f:
                f.write(f"{label.item()},{img_distance},"
                        f"{anomaly_score},{z_distance}\n")

# begin test!
test_anomaly_detection(generator, discriminator, encoder, test_dataloader, device)