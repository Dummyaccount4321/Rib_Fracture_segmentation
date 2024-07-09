#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from itertools import product
import logging
import torch.nn as nn
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
import skimage.transform as skTrans
import torch.nn.functional as F
import torch.nn as nn
from itertools import product
import nibabel as nib
import numpy as np
import torch
from skimage.measure import regionprops
from torch.utils.data import DataLoader, Dataset


# In[2]:


class Window:

    def __init__(self, window_min, window_max):
        self.window_min = window_min
        self.window_max = window_max

    def __call__(self, image):
        image = np.clip(image, self.window_min, self.window_max)

        return image


# In[3]:


class MinMaxNorm:

    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self, image):
        image = (image - self.low) / (self.high - self.low)
        image = image * 2 - 1

        return image


# In[4]:


class FracNetInferenceDataset(Dataset):

    def __init__(self, image_path, crop_size=128, transforms=None):
        self.crop_size = crop_size
        image = nib.load(image_path)
        self.image_affine = image.affine
        image = image.get_fdata().astype(np.int16)
        image = self._add_slices(image)
        self.image = image
        self.transforms = transforms
        self.centers = self._get_centers()

    def _get_centers(self):
        dim_coords = [list(range(0, dim, self.crop_size // 2))[1:-1]\
            + [dim - self.crop_size // 2] for dim in self.image.shape]
        centers = list(product(*dim_coords))

        return centers

    def __len__(self):
        return len(self.centers)

    def _crop_patch(self, idx):
        center_x, center_y, center_z = self.centers[idx]
        patch = self.image[
            center_x - self.crop_size // 2:center_x + self.crop_size // 2,
            center_y - self.crop_size // 2:center_y + self.crop_size // 2,
            center_z - self.crop_size // 2:center_z + self.crop_size // 2
        ]

        return patch

    def _apply_transforms(self, image):
        for t in self.transforms:
            image = t(image)

        return image
        
    def _add_slices(self, image):
        if image.shape[2] < self.crop_size:
            num_slices_to_add = self.crop_size - image.shape[2]
            pad_width = ((0, 0), (0, 0), (0, num_slices_to_add))
            image = np.pad(image, pad_width, mode='constant', constant_values=0)

        return image

 

    def __getitem__(self, idx):
        image = self._crop_patch(idx)
        center = self.centers[idx]

        if self.transforms is not None:
            image = self._apply_transforms(image)
            


        image = torch.tensor(image[np.newaxis], dtype=torch.float)

            

        
        return image, center

    @staticmethod
    def _collate_fn(samples):
        images = torch.stack([x[0] for x in samples])
        centers = [x[1] for x in samples]

        return images,centers

    @staticmethod
    def get_dataloader(dataset, batch_size, num_workers=0):
        return DataLoader(dataset, batch_size, num_workers=num_workers,
            collate_fn=FracNetInferenceDataset._collate_fn)


# In[5]:


class MixLoss(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x, y):
        lf, lfw = [], []
        for i, v in enumerate(self.args):
            if i % 2 == 0:
                lf.append(v)
            else:
                lfw.append(v)
        mx = sum([w * l(x, y) for l, w in zip(lf, lfw)])
        return mx
    
class DiceLoss(nn.Module):
    def __init__(self, image=False):
        super().__init__()
        self.image = image

    def forward(self, x, y):
        x = x.sigmoid()
        i, u = [t.flatten(1).sum(1) if self.image else t.sum() for t in [x * y, x + y]]

        dc = (2 * i + 1) / (u + 1)
        dc = 1 - dc.mean()
        return dc

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
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, x, y):
        ce = F.binary_cross_entropy_with_logits(x, y)
        fc = self.alpha * (1 - torch.exp(-ce)) ** self.gamma * ce
        return fc


# In[6]:


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


# In[7]:


import os
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import ndimage
from skimage.measure import label, regionprops
from skimage.morphology import disk, remove_small_objects
from tqdm import tqdm


# In[8]:


def _remove_low_probs(pred, prob_thresh):
    pred = np.where(pred > prob_thresh, pred, 0)

    return pred

def _remove_spine_fp(pred, image, bone_thresh):
    image_bone = image > bone_thresh
    image_bone_2d = image_bone.sum(axis=-1)
    image_bone_2d = ndimage.median_filter(image_bone_2d, 10)
    image_spine = (image_bone_2d > image_bone_2d.max() // 3)
    kernel = disk(7)
    image_spine = ndimage.binary_opening(image_spine, kernel)
    image_spine = ndimage.binary_closing(image_spine, kernel)
    image_spine_label = label(image_spine)
    max_area = 0

    for region in regionprops(image_spine_label):
        if region.area > max_area:
            max_region = region
            max_area = max_region.area
    image_spine = np.zeros_like(image_spine)
    image_spine[
        max_region.bbox[0]:max_region.bbox[2],
        max_region.bbox[1]:max_region.bbox[3]
    ] = max_region.convex_image > 0

    return np.where(image_spine[..., np.newaxis], 0, pred)

def _remove_small_objects(pred, size_thresh):
    pred_bin = pred > 0
    pred_bin = remove_small_objects(pred_bin, size_thresh)
    pred = np.where(pred_bin, pred, 0)

    return pred


def _post_process(pred, image, prob_thresh, bone_thresh, size_thresh):

    # remove connected regions with low confidence
    pred = _remove_low_probs(pred, prob_thresh)

    # remove spine false positives
    pred = _remove_spine_fp(pred, image, bone_thresh)

    # remove small connected regions
    pred = _remove_small_objects(pred, size_thresh)
    
    return pred


# In[9]:


def _predict_single_image(model, dataloader, postprocess, prob_thresh,
        bone_thresh, size_thresh):
    pred = np.zeros(dataloader.dataset.image.shape)
    crop_size = dataloader.dataset.crop_size
    with torch.no_grad():
        for _, sample in enumerate(dataloader):
            images, centers = sample
            images = images.cuda()

            output = model(images)[0].sigmoid().cpu().numpy().squeeze(axis=1)

            for i in range(len(centers)):
                center_x, center_y, center_z = centers[i]
                cur_pred_patch = pred[
                    center_x - crop_size // 2:center_x + crop_size // 2,
                    center_y - crop_size // 2:center_y + crop_size // 2,
                    center_z - crop_size // 2:center_z + crop_size // 2
                ]
                pred[
                    center_x - crop_size // 2:center_x + crop_size // 2,
                    center_y - crop_size // 2:center_y + crop_size // 2,
                    center_z - crop_size // 2:center_z + crop_size // 2
                ] = np.where(cur_pred_patch > 0, np.mean((output[i],
                    cur_pred_patch), axis=0), output[i])

    if postprocess:
        pred = _post_process(pred, dataloader.dataset.image, prob_thresh,
            bone_thresh, size_thresh)
    
    return pred


# In[10]:


def _make_submission_files(pred, image_id, affine):
    pred_label = label(pred > 0).astype(np.int16)
    pred_regions = regionprops(pred_label, pred)
    pred_index = [0] + [region.label for region in pred_regions]
    pred_proba = [0.0] + [region.mean_intensity for region in pred_regions]
    # placeholder for label class since classifaction isn't included
    pred_label_code = [0] + [1] * int(pred_label.max())
    pred_image = nib.Nifti1Image(pred_label, affine)
    pred_info = pd.DataFrame({
        "public_id": [image_id] * len(pred_index),
        "label_id": pred_index,
        "confidence": pred_proba,
        "label_code": pred_label_code
    })

    return pred_image, pred_info


# In[11]:


def predict(args):
    batch_size = 8
    num_workers = 4
    postprocess = args['postprocess']
    model = UNet(1, 1, first_out_channels=16)
    model.eval()
    if args['model_path'] is not None:
        model_weights = torch.load(args['model_path'])
        if 'model' in model_weights:
            model_weights = model_weights['model']
            model.load_state_dict(model_weights)
        else:
            model.load_state_dict(model_weights)
    model = nn.DataParallel(model).cuda()

    transforms = [
        Window(-200, 1000),
        MinMaxNorm(-200, 1000)
    ]
    image_path_list = sorted([os.path.join(args['image_dir'], file)
        for file in os.listdir(args['image_dir']) if "nii" in file])
    image_id_list = [os.path.basename(path).split("-")[0]
        for path in image_path_list]

    progress = tqdm(total=len(image_id_list))
    pred_info_list = []
    for image_id, image_path in zip(image_id_list, image_path_list):
        dataset = FracNetInferenceDataset(image_path, transforms=transforms)
        dataloader = FracNetInferenceDataset.get_dataloader(dataset,
            batch_size, num_workers)
        pred_arr = _predict_single_image(model, dataloader, postprocess,
            args['prob_thresh'], args['bone_thresh'], args['size_thresh'])
        pred_image, pred_info = _make_submission_files(pred_arr, image_id,
            dataset.image_affine)
        pred_info_list.append(pred_info)
        pred_path = os.path.join(args['pred_dir'], f"{image_id}_pred.nii.gz")
        nib.save(pred_image, pred_path)

        progress.update()

    pred_info = pd.concat(pred_info_list, ignore_index=True)
    pred_info.to_csv(os.path.join(args['pred_dir'], "pred_info.csv"),
        index=False)


# In[ ]:


args = {'image_dir' :'/workspace/ribfrac/Ribfrac-Dataset/Test2/ribfrac-test-images',
       'pred_dir':'./ribfrac_test_0.6_100_3',
       'model_path' :'./bestmodel_chunet.pth',
       'prob_thresh' :0.6,
       'bone_thresh': 100,
       'size_thresh': 150,
       'postprocess' : True}
predict(args)


# In[ ]:





# In[ ]:




