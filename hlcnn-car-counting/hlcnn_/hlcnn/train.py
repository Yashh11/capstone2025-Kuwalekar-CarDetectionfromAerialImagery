import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib as mpl
from matplotlib import pyplot as plot
from CARPK import CARPK
from PUCPR import PUCPR
import localizerVgg
from scipy.ndimage.filters import maximum_filter, median_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

#peak detection utility on a heatmap
def detect_peaks(image):
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    background = (image == 0)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
    detected_peaks = local_max ^ eroded_background
    return detected_peaks

#loss: mae btw predicted vs target GAMs
class nllloss(nn.Module):
    def __init__(self):
        super(nllloss, self).__init__()

    def forward(self, y_pred, y, num_car):
        y_pred = y_pred.view(y_pred.shape[0], -1)
        y = y.view(y.shape[0], -1)
        y_pred = torch.abs(y_pred - y.float())
        ret = torch.sum(y_pred) / (y_pred.shape[0]*y_pred.shape[1])
        return ret

#model setup
model = localizerVgg.localizervgg16(pretrained=True)
model.cuda()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

train_dataset = CARPK('datasets', 'train', train=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
criterionGAM = nllloss()
optimizer_ft = optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)
model.train()

#epoch
for epoch in range(35):
    scheduler.step(epoch)
    for batch_idx, (data, GAM, numCar) in enumerate(train_loader):
        data, GAM, numCar = data.to(device, dtype=torch.float),  GAM.to(device), numCar.to(device)
        MAP = model(data)
        cMap = MAP[0,0,].data.cpu().numpy()
        cMap = (cMap - cMap.min()) / (cMap.max() - cMap.min())
        cMap[cMap < 0.1] = 0
        peakMAP = detect_peaks(cMap)

        MAP = MAP.view(MAP.shape[0], -1)
        GAM = GAM.view(GAM.shape[0], -1)

        fark = abs(np.sum(peakMAP) - int(numCar[0]))

        loss = criterionGAM(MAP, GAM, numCar)

        optimizer_ft.zero_grad()
        loss.backward()
        optimizer_ft.step()

        if batch_idx % 20 == 0:
            print('Epoch: [{0}][{1}/{2}]\t' 'Loss: {3}\ AE:{4}'
                 .format(epoch, batch_idx, len(train_loader), loss,  abs(fark)))

    torch.save(model.state_dict(), 'trained_model')
