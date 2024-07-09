
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"


# In[2]:


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

import numpy as np

from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F


# In[3]:


class FracNetTrainDataset(Dataset):

    def __init__(self, image_dir, label_dir,img_save_dir, label_save_dir , crop_size=128,
            transforms=None, num_samples=18, train=True):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.public_id_list = sorted([x.split("-")[0]
            for x in os.listdir(image_dir)])
        self.crop_size = crop_size
        self.transforms = transforms
        self.num_samples = num_samples
        self.train = train
        self.img_save_dir = img_save_dir
        self.label_save_dir = label_save_dir
        self.current_number = 1

    def __len__(self):
        return len(self.public_id_list)

    @staticmethod
    def _get_pos_centroids(label_arr):
        centroids = [tuple([round(x) for x in prop.centroid])
            for prop in regionprops(label_arr)]

        return centroids
        
    def _get_augmented_pos_centroids(self,pos_centroids):
        
        augmented_centroids = []
        for centroid in pos_centroids:
            shifted_centroid = list(centroid)
            for axis in range(3):
                shift_amount = np.random.randint(-self.crop_size // 2, self.crop_size // 2)
                shifted_centroid[axis] += shift_amount
            augmented_centroids.append(tuple(shifted_centroid))
    
        return augmented_centroids

    @staticmethod
    def _get_symmetric_neg_centroids(pos_centroids, x_size):
        sym_neg_centroids = [(x_size - x, y, z) for x, y, z in pos_centroids]

        return sym_neg_centroids

    @staticmethod
    def _get_spine_neg_centroids(shape, crop_size, num_samples):
        x_min, x_max = shape[0] // 2 - 40, shape[0] // 2 + 40
        y_min, y_max = 300, 400
        z_min, z_max = crop_size // 2, shape[2] - crop_size // 2
        spine_neg_centroids = [(
            np.random.randint(x_min, x_max),
            np.random.randint(y_min, y_max),
            np.random.randint(z_min, z_max)
        ) for _ in range(num_samples)]

        return spine_neg_centroids
    
    @staticmethod
    def _get_random_centroids(label_arr,num_samples,other_neg_centroids,crop_size=64):
        random_neg_centroids = []
        max_attempts = 20  

        for _ in range(num_samples):
            attempts = 0
            while attempts < max_attempts:
                x = np.random.randint(crop_size // 2, label_arr.shape[0] - crop_size // 2)
                y = np.random.randint(crop_size // 2, label_arr.shape[1] - crop_size // 2)
                z = np.random.randint(crop_size // 2, label_arr.shape[2] - crop_size // 2)

                # Check if the random centroid overlaps with positive regions or other negative centroids
                overlapping_with_pos = np.any(label_arr[x - crop_size // 2:x + crop_size // 2,
                                                    y - crop_size // 2:y + crop_size // 2,
                                                    z - crop_size // 2:z + crop_size // 2] > 0)
                overlapping_with_other_neg = any(
                    np.linalg.norm(np.array((x, y, z)) - np.array(centroid)) < crop_size
                    for centroid in other_neg_centroids
                )

                if not overlapping_with_pos and not overlapping_with_other_neg:
                    random_neg_centroids.append((x, y, z))
                    break
                else:
                    attempts += 1

        return random_neg_centroids
    
    def _get_neg_centroids(self, pos_centroids, image_shape, label_arr):
        num_pos = len(pos_centroids)
        
        #1/3rd from symmentric region
        num_symmentric_samples = num_pos//3
        pos_centroids_symm = [pos_centroids[i] for i in np.random.choice(
                    range(0, len(pos_centroids)), size=num_symmentric_samples, replace=False)]
        sym_neg_centroids = self._get_symmetric_neg_centroids(
            pos_centroids_symm, image_shape[0])

        #1/3rd from spine region
        num_spine_centeroids = num_pos//3
        spine_neg_centroids = self._get_spine_neg_centroids(image_shape,self.crop_size,num_spine_centeroids)
        
        non_rand_centroids = sym_neg_centroids + spine_neg_centroids
        
        #1/3rd from random region 
        num_rand_centeroids = num_pos - num_symmentric_samples - num_spine_centeroids
        rand_centeroids = self._get_random_centroids(label_arr,num_rand_centeroids,non_rand_centroids ,crop_size=64)
        
        return non_rand_centroids + rand_centeroids

    def _get_roi_centroids(self, label_arr):
        if self.train:
            # generate positive samples' centroids
            pos_centroids = self._get_pos_centroids(label_arr)
            num_pos = len(pos_centroids)
            aug_pos_centroids = self._get_augmented_pos_centroids(pos_centroids)
            pos_centroids = pos_centroids + aug_pos_centroids
#             print(len(pos_centroids))
#             # generate negative samples' centroids
#             neg_centroids = self._get_neg_centroids(pos_centroids,
#                 label_arr.shape, label_arr)
#             print(len(neg_centroids))
# #             print(f'pos_centroids:{len(pos_centroids)} neg_centroids:{len(neg_centroids)}')
            roi_centroids = pos_centroids 
#     + neg_centroids
        else:
            roi_centroids = [list(range(0, x, y // 2))[1:-1] + [x - y // 2]
                for x, y in zip(label_arr.shape, self.crop_size)]
            roi_centroids = list(product(*roi_centroids))

        roi_centroids = [tuple([int(x) for x in centroid]) for centroid in roi_centroids]
        # print(f"ROI_centroids{type(roi_centroids)}")
        roi_centroids = random.sample(roi_centroids, len(roi_centroids))
        
        return roi_centroids

    def _crop_roi(self, arr, centroid):
        roi = np.ones(tuple([self.crop_size] * 3)) * (-1024)

        src_beg = [max(0, centroid[i] - self.crop_size // 2)
            for i in range(len(centroid))]
        src_end = [min(arr.shape[i], centroid[i] + self.crop_size // 2)
            for i in range(len(centroid))]
        dst_beg = [max(0, self.crop_size // 2 - centroid[i])
            for i in range(len(centroid))]
        dst_end = [min(arr.shape[i] - (centroid[i] - self.crop_size // 2),
            self.crop_size) for i in range(len(centroid))]
        roi[
            dst_beg[0]:dst_end[0],
            dst_beg[1]:dst_end[1],
            dst_beg[2]:dst_end[2],
        ] = arr[
            src_beg[0]:src_end[0],
            src_beg[1]:src_end[1],
            src_beg[2]:src_end[2],
        ]

        return roi

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

        
        # calculate rois' centroids
        roi_centroids = self._get_roi_centroids(label_arr)

        # crop rois
        image_rois = [self._crop_roi(image_arr, centroid)
            for centroid in roi_centroids]
        label_rois = [self._crop_roi(label_arr, centroid)
            for centroid in roi_centroids]
        
        
        for i, (image_roi, label_roi) in enumerate(zip(image_rois, label_rois)):
            # save input patch (image) to file with continuous number
            image_filename = f"Ribpatch{self.current_number}-image.nii.gz"
            image_save_path = os.path.join(self.img_save_dir, image_filename)
            nib.save(nib.Nifti1Image(image_roi, image.affine), image_save_path)

            # save output patch (label) to file with continuous number
            label_filename = f"Ribpatch{self.current_number}-label.nii.gz"
            label_save_path = os.path.join(self.label_save_dir, label_filename)
            nib.save(nib.Nifti1Image(label_roi, label.affine), label_save_path)

            self.current_number += 1

        
        if self.transforms is not None:
            image_rois = [self._apply_transforms(image_roi)
                for image_roi in image_rois]
        image_rois = torch.tensor(np.stack(image_rois)[:, np.newaxis],
            dtype=torch.float)
        label_rois = (np.stack(label_rois) > 0).astype(np.float64)
        label_rois = torch.tensor(label_rois[:, np.newaxis],
            dtype=torch.float)
        
        
        return image_rois, label_rois

    @staticmethod
    def collate_fn(samples):
        image_rois = torch.cat([x[0] for x in samples])
        label_rois = torch.cat([x[1] for x in samples])

        has_fracture = torch.tensor([(x > 0).any() for x in label_rois], dtype=torch.float)

        return image_rois, (label_rois, has_fracture)

    @staticmethod
    def get_dataloader(dataset, batch_size, shuffle=False, num_workers=0):
        return DataLoader(dataset, batch_size, shuffle,
            num_workers=num_workers, collate_fn=FracNetTrainDataset.collate_fn)


# In[4]:


img_save_dir = '/workspace/ribfrac/Harini/sample_set_detection/Train/images'
label_save_dir = '/workspace/ribfrac/Harini/sample_set_detection/Train/labels'

train_image_dir ='/workspace/ribfrac/Ribfrac-Dataset/Train/images'
train_label_dir ='/workspace/ribfrac/Ribfrac-Dataset/Train/labels'

batch_size = 1
num_workers = 0
ds_train = FracNetTrainDataset(train_image_dir, train_label_dir,img_save_dir,label_save_dir )
dl_train = FracNetTrainDataset.get_dataloader(ds_train, batch_size, False,
        num_workers)


# In[ ]:


i=0
for idx,(images,labels) in dl_train:
    i+=1
    print(i,end=' ')


# In[ ]:
