import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def Draw_ROC(file1,file2,file3,file4,file5,file6,file7,file8):
    data_vae=pd.read_csv(file1)
    data_vae=pd.DataFrame(data_vae)

    data_vaegan=pd.read_csv(file2)
    data_vaegan=pd.DataFrame(data_vaegan)

    data_vaewgan=pd.read_csv(file3)
    data_vaewgan=pd.DataFrame(data_vaewgan)

    data_vaewgangp=pd.read_csv(file4)
    data_vaewgangp=pd.DataFrame(data_vaewgangp)

    data_gnomaly=pd.read_csv(file5)
    data_gnomaly=pd.DataFrame(data_gnomaly)

    data_skipgnomaly=pd.read_csv(file6)
    data_skipgnomaly=pd.DataFrame(data_skipgnomaly)

    data_fanogan=pd.read_csv(file7)
    data_fanogan=pd.DataFrame(data_fanogan)

    data_our=pd.read_csv(file8)
    data_our=pd.DataFrame(data_our)


    fpr_vae,tpr_vae,thresholds = roc_curve(list(data_vae['labels']),
                                           list(data_vae['scores']))
    roc_auc_vae = auc(fpr_vae , tpr_vae)

    fpr_vaegan,tpr_vaegan,thresholds = roc_curve(list(data_vaegan['labels']),
                                           list(data_vaegan['scores']))
    roc_auc_vaegan = auc(fpr_vaegan , tpr_vaegan)

    fpr_vaewgan,tpr_vaewgan,thresholds = roc_curve(list(data_vaewgan['labels']),
                                           list(data_vaewgan['scores']))
    roc_auc_vaewgan = auc(fpr_vaewgan , tpr_vaewgan)

    fpr_vaewgangp,tpr_vaewgangp,thresholds = roc_curve(list(data_vaewgangp['labels']),
                                           list(data_vaewgangp['scores']))
    roc_auc_vaewgangp = auc(fpr_vaewgangp , tpr_vaewgangp)

    fpr_gnomaly,tpr_gnomaly,thresholds = roc_curve(list(data_gnomaly['labels']),
                                           list(data_gnomaly['scores']))
    roc_auc_gnomaly = auc(fpr_gnomaly , tpr_gnomaly)

    fpr_skipgnomaly,tpr_skipgnomaly,thresholds = roc_curve(list(data_skipgnomaly['labels']),
                                           list(data_skipgnomaly['scores']))
    roc_auc_skipgnomaly = auc(fpr_skipgnomaly , tpr_skipgnomaly)

    fpr_fanogan,tpr_fanogan,thresholds = roc_curve(list(data_fanogan['labels']),
                                           list(data_fanogan['scores']))
    roc_auc_fanogan = auc(fpr_fanogan , tpr_fanogan)

    fpr_our,tpr_our,thresholds = roc_curve(list(data_our['labels']),
                                           list(data_our['scores']))
    roc_auc_our = auc(fpr_our , tpr_our)


    plt.figure(figsize=(10,8), dpi=300)
    lw = 2
    plt.plot(fpr_vae, tpr_vae,
             label='AUC of VAE = {0:0.2f}'
                   ''.format(roc_auc_vae),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr_vaegan, tpr_vaegan,
             label='AUC of VAE-GAN = {0:0.2f}'
                   ''.format(roc_auc_vaegan),
             color='navy', linestyle=':', linewidth=4)

    plt.plot(fpr_vaewgan, tpr_vaewgan,
             label='AUC of VAE-WGAN = {0:0.2f}'
                   ''.format(roc_auc_vaewgan),
             color='darkorange', linestyle=':', linewidth=4)

    plt.plot(fpr_vaewgangp, tpr_vaewgangp,
             label='AUC of VAE-WGANGP = {0:0.2f}'
                   ''.format(roc_auc_vaewgangp),
             color='cornflowerblue', linestyle=':', linewidth=4)

    plt.plot(fpr_gnomaly, tpr_gnomaly,
             label='AUC of GANomaly = {0:0.2f}'
                   ''.format(roc_auc_gnomaly),
             color='gold', linestyle=':', linewidth=4)

    plt.plot(fpr_skipgnomaly, tpr_skipgnomaly,
             label='AUC of Skip-GANomaly = {0:0.2f}'
                   ''.format(roc_auc_skipgnomaly),
             color='green', linestyle=':', linewidth=4)

    plt.plot(fpr_fanogan, tpr_fanogan,
             label='AUC of f-AnoGAN = {0:0.2f}'
                   ''.format(roc_auc_fanogan),
             color='blueviolet', linestyle=':', linewidth=4)

    plt.plot(fpr_our, tpr_our,
             label='AUC of Geo-Hgan = {0:0.2f}'
                   ''.format(roc_auc_our),
             color='darkred', linestyle='-', linewidth=4)
    # plt.plot(fpr_vae,tpr_vae,
    #          label='AUC VAE = %0.2f'% roc_auc_vae,)
    # plt.plot(fpr_vaegan,tpr_vaegan,'blue',label='VAE-GAN = %0.3f'% roc_auc_vaegan)
    # plt.plot(fpr_vaewgan,tpr_vaewgan,'blue',label='VAE-WGAN = %0.3f'% roc_auc_vaewgan)
    # plt.plot(fpr_vaewgangp,tpr_vaewgangp,'blue',label='VAE-WGANGP = %0.3f'% roc_auc_vaewgangp)
    # plt.plot(fpr_gnomaly,tpr_gnomaly,'blue',label='Gnomaly = %0.3f'% roc_auc_gnomaly)
    # plt.plot(fpr_skipgnomaly,tpr_skipgnomaly,'blue',label='Skip-Gnomaly = %0.3f'% roc_auc_skipgnomaly)
    # plt.plot(fpr_fanogan,tpr_fanogan,'blue',label='f-AnoGAN = %0.3f'% roc_auc_fanogan)
    # plt.plot(fpr_our,tpr_our,'blue',label='Geo-Hgan = %0.3f'% roc_auc_our)
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show(block=True)

Draw_ROC('/mnt/3.6T-DATA/CBN/Mining/Data_paper/ROC/dim39zheng/VAE.csv',
             '/mnt/3.6T-DATA/CBN/Mining/Data_paper/ROC/dim39zheng/VAE-GAN.csv',
             '/mnt/3.6T-DATA/CBN/Mining/Data_paper/ROC/dim39zheng/VAE-WGAN.csv',
             '/mnt/3.6T-DATA/CBN/Mining/Data_paper/ROC/dim39zheng/VAE-WGANGP.csv',
             '/mnt/3.6T-DATA/CBN/Mining/Data_paper/ROC/dim39zheng/gnomaly.csv',
             '/mnt/3.6T-DATA/CBN/Mining/Data_paper/ROC/dim39zheng/skipgnomaly.csv',
             '/mnt/3.6T-DATA/CBN/Mining/Data_paper/ROC/dim39zheng/fanogan.csv',
             '/mnt/3.6T-DATA/CBN/Mining/Data_paper/ROC/dim39zheng/our.csv')