import os

from functools import partial
from torch import save

from fastai.basic_train import Learner
from fastai.train import ShowGraph
from fastai.data_block import DataBunch
from torch import optim
from itertools import product
import logging

from fastai.callbacks import *
from fastai.vision import *
import torch
import torch.nn as nn

from itertools import product

import nibabel as nib
import numpy as np
import torch
from skimage.measure import regionprops
from torch.utils.data import DataLoader, Dataset

from functools import partial
from torch import save

from fastai.basic_train import Learner
from fastai.train import ShowGraph
from fastai.data_block import DataBunch
from torch import optim
from itertools import product
import logging
import torch.nn.functional as F
from fastai.callbacks import *
from fastai.vision import *

import torch
import torch.nn as nn

from itertools import product

import nibabel as nib
import numpy as np
import torch
from skimage.measure import regionprops
from torch.utils.data import DataLoader, Dataset

import numpy as np

from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['MixLoss', 'DiceLoss', 'GHMCLoss', 'FocalLoss']

from fastai.callback import Callback

class MixLoss(nn.Module):
    def __init__(self,alpha=None,gamma=2,is_class=True,linear_weights=None):
        super().__init__()
        self.f_loss = FocalLoss()
        self.mcc_loss = DiceLoss()
        self.is_class=is_class
        if is_class:
            
            self.bce_loss = BCE_classloss()
            self.linear_weights = linear_weights
            self.mse_loss = None
            
            if linear_weights is not None:
                self.mse_loss = MSE_Loss(linear_weights)
                
        self.epoch = 0
                
    
    
    def forward(self, x, y1, y2):
        f = self.f_loss(x[0],y1)
        d = self.mcc_loss(x[0],y1)
        c = self.bce_loss(x[1],y2)
        if self.epoch < 50:
            total_loss = f+d+0.5*c
        else:
            total_loss = f + d + (2/self.epoch)*c
        

        return total_loss

class MSE_Loss(nn.Module):
    def __init__(self,linear_weights):
        super().__init__()
        self.pool = nn.MaxPool3d(8)
        self.loss = nn.MSELoss()
        self.linear_weights = linear_weights

    def forward(self,x,y):
        new_weights = self.linear_weights.view(self.linear_weights[0],self.linear_weights[1],1,1,1)
        x = x*new_weights
        y = self.pool(y)
        diff = self.loss(x,y)
        return diff
        
        

class DiceLoss(nn.Module):
    def __init__(self, image=False):
        super().__init__()
        self.image = image

    def forward(self, x, y):
        
        #print("diceloss")
        x = x.sigmoid()
        i, u = [t.flatten(1).sum(1) if self.image else t.sum() for t in [x * y, x + y]]

        dc = (2 * i + 1) / (u + 1)
        dc = 1 - dc.mean()
        return dc
    
class MCC_Loss(nn.Module):
    def __init__(self,image=False):
        super(MCC_Loss, self).__init__()
        self.image = image

    def forward(self, x, y):
        x = x.sigmoid()
        tp, tn, fp, fn = [t.flatten(1).sum(1) if self.image else t.sum() for t in [x * y, (1-x)*(1-y),x*(1-y),(1-x)*y]]
        numerator = (tp*tn) - (fp*fn)
        denominator = ((tp+1+fp)* (tp+1+fn)* (tn+1+fp)* (tn+1+fn))**0.5
        # Adding 1 to the denominator to avoid divide-by-zero errors.
        mcc = numerator/(denominator+1)
        mcc = mcc.mean()
        return 1 - mcc
    
class GHMCLoss(nn.Module):
    def __init__(self, mmt=0, bins=10):
        super().__init__()
        self.mmt = mmt
        self.bins = bins
        self.edges = [x / bins for x in range(bins + 1)]
        self.edges[-1] += 1e-6

        if mmt > 0:
            self.acc_sum = [0] * bins

    def forward(self, x, y):
        
        w = torch.zeros_like(x)
        g = torch.abs(x.detach().sigmoid() - y)

        n = 0
        t = reduce(lambda x, y: x * y, w.shape)
        for i in range(self.bins):
            ix = (g >= self.edges[i]) & (g < self.edges[i + 1]); nb = ix.sum()
            if nb > 0:
                if self.mmt > 0:
                    self.acc_sum[i] = self.mmt * self.acc_sum[i] + (1 - self.mmt) * nb
                    w[ix] = t / self.acc_sum[i]
                else:
                    w[ix] = t / nb
                n += 1
        if n > 0:
            w = w / n
        gc = F.binary_cross_entropy_with_logits(x, y, w, reduction='sum') / t
        return gc

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        if isinstance(alpha,(int,float)):
            self.alpha=torch.tensor([alpha,1-alpha])
        if isinstance(alpha,list):
            self.alpha=torch.tensor(alpha)
            
    def forward(self, x, y):
        
        loss=F.binary_cross_entropy_with_logits(x,y,reduction='none')
        
        p = torch.sigmoid(x)
    
        pt = p * y + (1 - p) * (1 - y)
        focal_weight = (1 - pt) ** self.gamma
        
        if self.alpha is not None:
            
            self.alpha=self.alpha.to(y.device)
            loss=self.alpha[y.long()]*loss
            
        focal_loss = focal_weight * loss
        f_loss = focal_loss.mean()
        
        return f_loss



class BCE_classloss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super().__init__()
        
            
    def forward(self, x, y):

        loss=F.binary_cross_entropy_with_logits(x,y)
        
        return loss

class Window:

    def __init__(self, window_min, window_max):
        self.window_min = window_min
        self.window_max = window_max

    def __call__(self, image):
        image = np.clip(image, self.window_min, self.window_max)

        return image


class MinMaxNorm:

    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self, image):
        image = (image - self.low) / (self.high - self.low)
        image = image * 2 - 1

        return image

def dice(x, y,z, image=False):
    x=x[0]
    x = x.sigmoid()
    i, u = [t.flatten(1).sum(1) if image else t.sum() for t in [x * y, x + y]]
    dc = ((2 * i + 1) / (u + 1)).mean()
    return dc


def recall(x, y, z, thresh=0.1):
    x=x[0]
    # y=y[0]
    x = x.sigmoid()
    tp = (((x * y) > thresh).flatten(1).sum(1) > 0).sum()
    rc = tp / (((y > 0).flatten(1).sum(1) > 0).sum() + 1e-8)
    return rc





def precision(x, y,z, thresh=0.1):
    x=x[0]
    # y=y[0]
    x = x.sigmoid()
    tp = (((x * y) > thresh).flatten(1).sum(1) > 0).sum()
    pc = tp / (((x > thresh).flatten(1).sum(1) > 0).sum() + 1e-8)
    return pc

def fbeta_score(x, y,z,beta=1, **kwargs):
    # x=x[0]
    # y=y[0]
    rc = recall(x, y, z,**kwargs)
    pc = precision(x, y, z, **kwargs)
    fs = (1 + beta ** 2) * pc * rc / (beta ** 2 * pc + rc + 1e-8)
    return fs

def class_score(x,y,z, **kwargs):
    x=x[1].sigmoid()
    x = (x > 0.5).long()
    correct = (x == z).sum()
    total = x.size(0)
    accuracy = (correct / total) 
    return accuracy




# In[6]:


class FracNetTrainDataset(Dataset):

    def __init__(self, image_dir, label_dir, transforms=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.public_id_list = sorted([x.split("-")[0]
            for x in os.listdir(image_dir)])
        self.transforms = transforms

    def __len__(self):
        return len(self.public_id_list)

    def _apply_transforms(self, image):
        for t in self.transforms:
            image = t(image)

        return image

    def __getitem__(self, idx):
        # read image and label
        public_id = self.public_id_list[idx]
        image_path = os.path.join(self.image_dir, f"{public_id}-image.nii.gz")
        label_path = os.path.join(self.label_dir, f"{public_id}-label.nii.gz")
        image = nib.load(image_path)
        label = nib.load(label_path)
        image_arr = image.get_fdata().astype(np.float64)
        label_arr = label.get_fdata().astype(np.uint8)
        label_arr[label_arr > 0] = 1


        if self.transforms is not None:
            image_arr = self._apply_transforms(image_arr)
                
        image_arr = torch.tensor(image_arr[np.newaxis, ...],dtype=torch.float)
        label_arr = torch.tensor(label_arr[np.newaxis,...],dtype=torch.float)
        class_label = torch.tensor(int(torch.any(label_arr > 0)),dtype=torch.float)

        class_label = class_label[np.newaxis]
        return image_arr, [label_arr, class_label]
        
    @staticmethod
    def get_dataloader(dataset, batch_size, shuffle=True, num_workers=0):
        return DataLoader(dataset, batch_size, shuffle, num_workers=num_workers)


# In[7]:
class UNet(nn.Module):
    def __init__(self, in_channels, num_classes, first_out_channels=16):
        super().__init__()
        self.first = ConvBlock(in_channels, first_out_channels)
        in_channels = first_out_channels
        self.down1 = Down(in_channels, 2 * in_channels)
        self.down2 = Down(2 * in_channels, 4 * in_channels)
        self.down3 = Down(4 * in_channels, 8 * in_channels)
        self.up1   = Up(8 * in_channels, 4 * in_channels)
        self.up2   = Up(4 * in_channels, 2 * in_channels)
        self.up3   = Up(2 * in_channels, in_channels)
        

        self.final = nn.Conv3d(in_channels, num_classes, 1)
        
        
        self.pool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.leaky_relu = nn.LeakyReLU(inplace=True)

        self.l1 = nn.Linear(8*in_channels ,1)


        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x1 = self.first(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # x =c3
        c3 = self.pool(x4)
        c3 = c3.view(c3.size(0), -1)
        z = self.l1(c3)

        a = self.l1.weight.view(1, -1, 1, 1, 1)
        s = x4 * a
        s = F.sigmoid(s)
        x4 = s * x4

        x5  = self.up1(x4, x3)
        x6  = self.up2(x5, x2)
        x7  = self.up3(x6, x1)
        
        
        y  = self.final(x7)
        
       
        


        return (y,z)
    

class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.MaxPool3d(2),
            ConvBlock(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x, y):
        x = self.conv2(x)
        x = self.conv1(torch.cat([y, x], dim=1))
        return x


# In[8]:


train_image_dir =  '/workspace/ribfrac/Harini/sample_set2/Train/images'
train_label_dir ='/workspace/ribfrac/Harini/sample_set2/Train/labels'
val_image_dir = '/workspace/ribfrac/Harini/sample_set2/Val/images'
val_label_dir ='/workspace/ribfrac/Harini/sample_set2/Val/labels'


batch_size = 16
num_workers = 4
optimizer = optim.Adam

thresh = 0.1
recall_partial = partial(recall, thresh=thresh)
precision_partial = partial(precision, thresh=thresh)
fbeta_score_partial = partial(fbeta_score, thresh=thresh)
class_score_partial = partial(class_score)

model = UNet(1, 1, first_out_channels=16)
model = nn.DataParallel(model.cuda())
alpha=[0.25,0.75]
gamma=2
criterion = MixLoss(alpha,gamma)
transforms = [
    Window(-200, 1000),
    MinMaxNorm(-200, 1000),
    
]
ds_train = FracNetTrainDataset(train_image_dir, train_label_dir,
        transforms=transforms)
dl_train = FracNetTrainDataset.get_dataloader(ds_train, batch_size, False,
        num_workers)
ds_val = FracNetTrainDataset(val_image_dir, val_label_dir,
        transforms=transforms)
dl_val = FracNetTrainDataset.get_dataloader(ds_val, batch_size, False,
        num_workers)



# In[9]:


databunch = DataBunch(dl_train, dl_val)




class ModelLosscallback(LearnerCallback):
    
    def __init__(self, learn:Learner):
        super().__init__(learn)
        
    def on_epoch_end(self, epoch:int, **kwargs):
        learn.loss_func.epoch=epoch
            



learn = Learner(
        databunch,
        model,
        opt_func=optimizer,
        loss_func=criterion,
        model_dir='model3',
        metrics=[dice, recall_partial, precision_partial, fbeta_score_partial, class_score_partial],
        callback_fns = [ModelLosscallback]
    )


learn.fit_one_cycle(
        100,
        1e-3,
        pct_start=0,
        div_factor=100,
        callbacks=[
            ShowGraph(learn),
            SaveModelCallback(learn)
        ]
    )







