import os
import requests

from app.kay_dload import net, best_model_wts, dat, mid_outputs_ft, mid_outputs_im
from config.settings import path_routing
from torch.utils.data import Dataset

import copy

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch import nn, optim
from torch.utils.data import Dataset
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter

from torchvision import transforms
import torchvision.models as models


from scipy.spatial.distance import pdist
from scipy.stats import pearsonr

# # load best model weights
# net.load_state_dict(best_model_wts)
#
# ## Extract features of all the intermediate layers from ImageNet-trained and finetuned Alexnet
# return_layers = {
#     'conv1': 'conv1',
#     'layer1': 'layer1',
#     'layer2': 'layer2',
#     'layer3': 'layer3',
#     'layer4': 'layer4',
#     'fc': 'fc',
# }
#
# # Loading resnet pretrained on Imagenet
# net_im = models.resnet18(pretrained=True)
# net_im.eval()
# net_im.to(device)
#
# # Setting up feature extraction step
# midfeat_ft = MidGetter(net, return_layers=return_layers, keep_output=True)
# midfeat_im = MidGetter(net_im, return_layers=return_layers, keep_output=True)
#
# # Loading validation data and forward pass through the network
# dataloaders = {x: torch.utils.data.DataLoader(dataset[x], batch_size=120) for x in ['val']}
# for inputs, labels in dataloaders['val']:
#     inputs = inputs.to(device)
#     mid_outputs_ft, _ = midfeat_ft(inputs)
#     mid_outputs_im, _ = midfeat_im(inputs)

# @title Dissimilarity - Correlation
# Loading V1 and LOC responses
v1_id = np.where(dat['roi'] == 1)
# loc_id = np.where(dat['roi'] == 7)
Rts_v1 = np.squeeze(dat["responses_test"][:, v1_id])
# Rts_lo = np.squeeze(dat["responses_test"][:, loc_id])

# Observed dissimilarity  - Correlation
fMRI_dist_metric_ft = "euclidean"  # ['correlation', 'euclidean']
fMRI_dist_metric_im = "correlation"  # ['correlation', 'euclidean']

CNN_ft_dist_metric = "euclidean"  # ['correlation', 'euclidean']
CNN_im_dist_metric = "correlation"  # ['correlation', 'euclidean']

dobs_v1_ft = pdist(Rts_v1, fMRI_dist_metric_ft)
# dobs_lo_ft = pdist(Rts_lo, fMRI_dist_metric_ft)
dobs_v1_im = pdist(Rts_v1, fMRI_dist_metric_im)
# dobs_lo_im = pdist(Rts_lo, fMRI_dist_metric_im)

# Comparing representation of V1 and LOC across different layers of Alexnet
r, p = np.zeros((4, 8)), np.zeros((4, 8))
for i,l in enumerate(mid_outputs_ft.keys()):
    dnet_ft = pdist(torch.flatten(mid_outputs_ft[l], 1, -1).cpu().detach().numpy(),
                  CNN_ft_dist_metric)
    dnet_im = pdist(torch.flatten(mid_outputs_im[l], 1, -1).cpu().detach().numpy(),
                  CNN_im_dist_metric)
    r[0, i], p[0, i] = pearsonr(dnet_ft, dobs_v1_ft)
    r[1, i], p[1, i] = pearsonr(dnet_im, dobs_v1_im)
    # r[2, i], p[2, i] = pearsonr(dnet_ft, dobs_lo_ft)
    # r[3, i], p[3, i] = pearsonr(dnet_im, dobs_lo_im)

# @title Plotting correlation between observed and predicted dissimilarity values
plt.bar(range(6), r[0, :], alpha=0.5)
plt.bar(range(6), r[1, :], alpha=0.5)
plt.legend(['Fine Tuned', 'Imagenet'])
plt.ylabel('Correlation coefficient')
plt.title('Match to V1')
plt.xticks(range(6), mid_outputs_ft.keys())
plt.show()

# plt.figure()
# plt.bar(range(6), r[2, :], alpha=0.5)
# plt.bar(range(6), r[3, :], alpha=0.5)
# plt.legend(['Fine Tuned', 'Imagenet'])
# plt.ylabel('Correlation coefficient')
# plt.title('Match to LOC')
# plt.xticks(range(6), mid_outputs_ft.keys())
# plt.show()



