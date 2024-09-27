#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 11:33:08 2024

@author: des
"""

import torchio as tio
import skimage.filters
import skimage.measure
import skimage.segmentation

from folder_locations import get_data_folder

def replace_zero_background(img):
    arr = img.numpy()
    arr = skimage.segmentation.flood_fill(arr, (0, 0, 0, 0), -1000)
    img.set_data(arr)

def remove_background(img):
    mask = tio.LabelMap(tensor = img.numpy() > 200)
    
    sub = tio.Subject(
        scan = img,
        mask = mask
        )
    ca_transform = tio.transforms.CopyAffine('scan')
    sub = ca_transform(sub)
    crop_transform = tio.transforms.Compose([
        tio.transforms.Resample(1, image_interpolation="BSPLINE"),
        tio.transforms.CropOrPad(196, mask_name="mask", padding_mode=0)
        ])
    sub = crop_transform(sub)
    img = sub.scan
    
    # Due to a bug in torchio resampling and padding inserts some zeros a the
    # edge of an image. Therefore we used zeros for padding and replace them
    #here with -1000
    replace_zero_background(img)
    
    return img
    

if __name__ == "__main__":
    model_path = get_data_folder() / "pretrained networks" / "resnet_10_23dataset.pth"
    images_path = get_data_folder() / "13.06.2024_XXXX_THE LAST CT SCANNING DATA"
    results_path =  get_data_folder() / "13.06.2024_cropped_CTs"
    (results_path / "injured").mkdir(parents = True, exist_ok=True)
    (results_path / "healthy").mkdir(exist_ok=True)
    
    sizes = []
    sorted_sizes = []
    
    spacings = {}
    for p in images_path.glob("**/*.nii.gz"):
        print(p)
        key = p.name[0]
        if key not in spacings.keys():
            spacings[key] = []
        
        img = tio.ScalarImage(p)
        new_img = remove_background(img)
        out_path = results_path / p.relative_to(images_path)
        new_img.save(out_path)
        