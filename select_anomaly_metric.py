'''
As three types of anomaly score calculation criteria, namely z_distance, img_distance, and anomaly_score, have been computed,
where anomaly_score is the weighted sum of z_distance and img_distance,
the purpose of this code is to determine which metric is most effective in identifying anomalies (based on the maximum AUC)
and to use this metric as the standard for calculating the anomaly scores of the samples.
'''
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from nets.sample_GAN import WGAN_Discriminator,WGAN_Generator,Encoder
from nets.autoencoder import Z_Encoder
from tools import loadTifImage
from skimage import io
import os
import pandas as pd
import numpy as np
from sklearn.metrics import auc, roc_curve


# Set the relevant hyperparameters
batch_size = 1
img_size = 64
latent_dim = 100
channels = 10

# Specify the GPU used for training
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

# load the test dataset
transform = transforms.Compose([transforms.ToTensor()])
testpath = "..." # the path of the test data (should be modified to your own path of the test dataset)
test_dataset = loadTifImage.DatasetFolder(root=testpath, transform=transform)
test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=4,drop_last=False)

# define the reconstructed loss
criterion = nn.MSELoss()

# sample adversarial generator
# G = WGAN_Generator(img_size=img_size, latent_dim=latent_dim, channels=channels).to(device)
# G.load_state_dict(torch.load("/mnt/3.6T-DATA/CBN/train_wgangp_latent/conv1204/G_wgangp.pth")) # pretrained weights saved by running "train_LSLM.py"
G = torch.load("/mnt/3.6T-DATA/CBN/train_wgangp_latent/dim39_attention/G_wgangp.pth")
G.eval()

# sample adversarial discriminator
# D = WGAN_Discriminator(channels=channels, img_size=img_size).to(device)
# D.load_state_dict(torch.load("/mnt/3.6T-DATA/CBN/train_wgangp_latent/conv1204/D_wgangp.pth")) # pretrained weights saved by running "train_LSLM.py"
D = torch.load("/mnt/3.6T-DATA/CBN/train_wgangp_latent/dim39_attention/D_wgangp.pth")
D.eval()

# encoder
E = Z_Encoder(img_shape=img_shape,latent_dim=latent_dim)
E.load_state_dict(torch.load("/mnt/3.6T-DATA/CBN/train_wgangp_latent/dim39_attention/encoderizif.pth")) # GAN-guided pretrained weights of the encoder

# the path for saving anomaly scores
score_path = "/mnt/3.6T-DATA/CBN/Data_paper/MPM/MPM_our.csv"
with open(score_path, "w") as f:
    f.write("label,img_distance,anomaly_score,z_distance\n")
for i,(imgs,labels) in enumerate(test_dataloader):
    real_imgs = imgs.to(device)

    real_z = E(real_imgs)
    fake_imgs = G(real_z)
    fake_z = E(fake_imgs)

    real_feature = D.forward_features(real_imgs)
    fake_feature = D.forward_features(fake_imgs)

    img_distance = criterion(fake_imgs, real_imgs)
    loss_feature = criterion(fake_feature, real_feature)
    anomaly_score = img_distance + loss_feature

    z_distance = criterion(fake_z, real_z)
    img_distance = img_distance * (10**10)
    anomaly_score = anomaly_score * (10**10)
    loss_feature = loss_feature * (10**10)

    with open(score_path, "a") as f:
        f.write(f"{labels.item()},{img_distance},"
                f"{anomaly_score},{z_distance}\n")

    # fake_savePath = "/mnt/3.6T-DATA/CBN/AnomalySamples/dim10ch"
    # os.makedirs(fake_savePath + "/img_distance", exist_ok=True)
    # os.makedirs(fake_savePath + "/anomaly_score", exist_ok=True)
    # os.makedirs(fake_savePath + "/z_distance", exist_ok=True)
    # save_img = fake_imgs.data.cpu().numpy().squeeze(0)
    # save_img = np.transpose(save_img,(1,2,0))
    # io.imsave(fake_savePath + "/img_distance/{}.tif".format(img_distance), save_img)
    # io.imsave(fake_savePath + "/anomaly_score/{}.tif".format(anomaly_score), save_img)
    # io.imsave(fake_savePath + "/z_distance/{}.tif".format(z_distance), save_img)


best_auc = 0
df_score = pd.read_csv(score_path)
trainig_label = 0
labels = np.where(df_score["label"].values == trainig_label, 0, 1)
anomaly_score = df_score["anomaly_score"].values
img_distance = df_score["img_distance"].values
z_distance = df_score["z_distance"].values
fpr_imgdist, tpr_imgdist, _ = roc_curve(labels,img_distance)
fpr_anomaly, tpr_anomaly, _ = roc_curve(labels, anomaly_score)
fpr_zdist, tpr_zdist, _ = roc_curve(labels, z_distance)
auc_imgdist = auc(fpr_imgdist, tpr_imgdist)
auc_anomaly = auc(fpr_anomaly, tpr_anomaly)
auc_zdist = auc(fpr_zdist, tpr_zdist)

selected_path = fake_savePath + "/img_distance/"
if auc_imgdist>=auc_anomaly and auc_imgdist>=auc_zdist:
    selected_path = fake_savePath + "/img_distance/"
    best_auc = auc_imgdist
elif auc_zdist>=auc_imgdist and auc_zdist>=auc_anomaly:
    selected_path = fake_savePath + "/z_distance/"
    best_auc = auc_zdist
elif auc_anomaly>=auc_imgdist and auc_anomaly>=auc_zdist:
    selected_path = fake_savePath + "/anomaly_score/"
    best_auc = auc_anomaly

print("Select the {} directory as the set of abnormal samples, with a maximum AUC of {}".format(selected_path,best_auc))









