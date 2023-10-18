import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.autograd as autograd
import torchvision.transforms as transforms
from skimage import io
from tools import loadTifImage
from nets.autoencoder import AE
from nets.latent_GAN import latent_Generator,latent_Discriminator
from nets.sample_GAN import WGAN_Discriminator,WGAN_Generator

# Specify the GPU used for training
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

'''
Set the relevant hyperparameters:
n_epochs: Number of epochs for training. 
batch_size: Batch size for training. 
lr: Learning rate for optimization. 
b1: The beta1 parameter used in the optimizer. 
b2: The beta2 parameter used in the optimizer.
latent_dim: Dimension of the latent space features.
img_size: Size of the samples.
channels: Channel of the samples.
n_critic: The number of times the critic (discriminator) is updated per generator update in the training process of a GAN.
sample_interval: Save generated samples every few iterations.
training_label: Since only normal data is used for training, here you should specify the index of the normal class.
lambda_gp: Coefficient of the gradient penalty term.
'''

n_epochs = 100
batch_size = 64
lr = 0.0002
b1 = 0.5
b2 = 0.999
latent_dim = 100
img_size = 64
channels = 39
n_critic = 5
sample_interval = 400
training_label = 0
lambda_gp = 10

# Load the training dataset (only normal (negative) data is used for training)
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = loadTifImage.DatasetFolder(root='/mnt/3.6T-DATA/CBN/DATA/dim39_zheng_anomaly/train/',
                                           transform=transform) # "root" should be modified to the path of your training dataset
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6, drop_last=True)


img_shape = (channels, img_size, img_size)
l_G = latent_Generator(z_dim=latent_dim) # latent space adversarial generator
l_D = latent_Discriminator(z_dim=latent_dim) # latent space adversarial discriminator
AE = AE(img_shape=(channels,img_size,img_size), latent_dim=latent_dim) # convolutional autoencoder
wgan_G = WGAN_Generator(img_size=img_size,latent_dim=latent_dim,channels=channels) # sample adversarial generator
wgan_D = WGAN_Discriminator(channels=channels,img_size=img_size) # sample adversarial discriminator

def compute_gradient_penalty(D, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(*real_samples.shape[:2], 1, 1, device=device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples)
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    d_interpolates = D(interpolates)
    fake = torch.ones(*d_interpolates.shape, device=device)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(outputs=d_interpolates, inputs=interpolates,
                              grad_outputs=fake, create_graph=True,
                              retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.shape[0], -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# train the latent space learning module
def train_wgangp(l_G, l_D, AE, wgan_G, wgan_D, train_dataloader, device, lambda_gp=10, n_epochs=n_epochs):
    d_losses = []
    g_losses = []

    l_G.to(device)
    l_D.to(device)
    AE.to(device)
    wgan_G.to(device)
    wgan_D.to(device)

    criterion_mse = nn.MSELoss() # reconstructed loss

    optimizer_wganG = torch.optim.Adam(wgan_G.parameters(), lr=lr, betas=(b1,b2)) # optimizer of sample adversarial generator
    optimizer_wganD = torch.optim.Adam(wgan_D.parameters(), lr=lr, betas=(b1,b2)) # optimizer of sample adversarial discriminator
    optimizer_AE = torch.optim.Adam(AE.parameters(), lr=lr, betas=(b1,b2)) # optimizer of AE
    optimizer_lG = torch.optim.Adam(l_G.parameters(), lr=lr, betas=(b1,b2)) # optimizer of latent space adversarial generator
    optimizer_lD = torch.optim.Adam(l_D.parameters(), lr=lr, betas=(b1,b2)) # optimizer of latent space adversarial discriminator

    # Create a folder to save the generated images. Please modify the path to your own directory.
    os.makedirs("/mnt/3.6T-DATA/CBN/train_wgangp_latent/dim39_attention" ,exist_ok=True) #
    os.makedirs("/mnt/3.6T-DATA/CBN/train_wgangp_latent/dim39_attention/generated_images_wgangp" ,exist_ok=True)

    padding_epoch = len(str(n_epochs))
    padding_i = len(str(len(train_dataloader)))
    batches_done = 0

    # Training loop
    for epoch in range(n_epochs):
        for i, (imgs,labels) in enumerate(train_dataloader):

            real_imgs = imgs.to(device)

            # ------------------- Optimizing the encoder and decoder of AE ------------------- #
            optimizer_AE.zero_grad()
            encoder_z, reconstructed_imgs = AE(real_imgs)
            loss_ae = criterion_mse(reconstructed_imgs, real_imgs)
            loss_ae.backward()
            optimizer_AE.step()

            # -------------------- Optimizing the latent space generative adversarial network -------------------- #
            # ----------------- Optimizing the latent space adversarial discriminator ----------------- #
            for _ in range(n_critic):
                optimizer_lD.zero_grad()
                encoder_z,_ = AE(real_imgs)
                random_noise = torch.randn(imgs.shape[0], latent_dim).to(device)
                latent_z = l_G(random_noise)
                real_z_validity = l_D(encoder_z)
                fake_z_validity = l_D(latent_z)
                latent_d_loss = (0.50 * torch.mean((real_z_validity - 1)**2)) + (0.50 * torch.mean((fake_z_validity)**2))
                latent_d_loss.backward()
                optimizer_lD.step()

            # ----------------- Optimizing the latent space adversarial generator ----------------- #
            optimizer_lG.zero_grad()
            random_noise = torch.randn(imgs.shape[0], latent_dim).to(device)
            latent_z = l_G(random_noise)
            fake_z_validity = l_D(latent_z)
            latent_g_loss = 0.50 * torch.mean((fake_z_validity-1)**2)
            latent_g_loss.backward()
            optimizer_lG.step()

            # -------------------- Optimizing the sample generative adversarial network -------------------- #
            # ----------------- Optimizing the sample adversarial discriminator ----------------- #
            optimizer_wganD.zero_grad()
            random_noise = torch.randn(imgs.shape[0], latent_dim).to(device)
            encoder_z, de_fake_imgs = AE(real_imgs)
            de_fake_validity = wgan_D(de_fake_imgs.detach())
            latent_z = l_G(random_noise)
            fake_imgs = wgan_G(latent_z)
            real_validity = wgan_D(real_imgs)
            fake_validity = wgan_D(fake_imgs.detach())
            gradient_penalty = compute_gradient_penalty(wgan_D, real_imgs.data, fake_imgs.data, device)
            wgan_d_loss = (-torch.mean(real_validity) + torch.mean(fake_validity) + 1e-3 * torch.mean(de_fake_validity) + lambda_gp * gradient_penalty)
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            wgan_d_loss.backward()
            optimizer_wganD.step()

            # ----------------- Optimizing the sample adversarial generator ----------------- #
            if i % n_critic == 0:
                optimizer_wganG.zero_grad()
                fake_imgs = wgan_G(latent_z)
                fake_validity = wgan_D(fake_imgs)
                wgan_g_loss = -torch.mean(fake_validity)
                wgan_g_loss.backward()
                optimizer_wganG.step()
                d_losses.append(wgan_d_loss.item())
                g_losses.append(wgan_g_loss.item())

                # Print the training process
                print(f"[Epoch {epoch:{padding_epoch}}/{n_epochs}] "
                      f"[Batch {i:{padding_i}}/{len(train_dataloader)}] "
                      f"[WGANGP_D loss: {wgan_d_loss.item():4f}] "
                      f"[WGANGP_G loss: {wgan_g_loss.item():4f}]"
                      f"[AE loss: {loss_ae.item():4f}]"
                      f"[l_G loss: {latent_g_loss.item():4f}]"
                      f"[l_D loss: {latent_d_loss.item():4f}]")
                
                # Save generated images every sample_interval iterations
                if batches_done % sample_interval == 0:
                    io.imsave(f"/mnt/3.6T-DATA/CBN/train_wgangp_latent/dim39_attention/generated_images_wgangp/{batches_done:06}.tif",
                              fake_imgs.data.cpu().numpy())
                batches_done += n_critic
                
    # save the model weights (the path should be modified to your own directory)
    torch.save(wgan_G,"/mnt/3.6T-DATA/CBN/train_wgangp_latent/dim39_attention/G_wgangp.pth")
    torch.save(wgan_D,"/mnt/3.6T-DATA/CBN/train_wgangp_latent/dim39_attention/D_wgangp.pth")
    torch.save(AE.encoder.state_dict(), "/mnt/3.6T-DATA/CBN/train_wgangp_latent/dim39_attention/encoder.pth")
    torch.save(l_G, "/mnt/3.6T-DATA/CBN/train_wgangp_latent/dim39_attention/l_G.pth")
    torch.save(l_D, "/mnt/3.6T-DATA/CBN/train_wgangp_latent/dim39_attention/l_D.pth")

# begin train!
train_wgangp(l_G,l_D,AE,wgan_G,wgan_D,train_dataloader,device,lambda_gp=10,n_epochs=n_epochs)

