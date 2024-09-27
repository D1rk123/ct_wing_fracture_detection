#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 15:08:33 2024

@author: des
"""

import torch
from torch import optim, utils
import torch.multiprocessing
import pytorch_lightning as pl
import torchio as tio
import timm_3d

import ct_experiment_utils as ceu
from folder_locations import get_results_folder, get_data_folder
from split_names import get_CT_split_names

class LCTClassificationModule(pl.LightningModule):
    def __init__(self, nr_classes, trial):
        super().__init__()
        self.nr_classes = nr_classes
        self.trial = trial
        
        self.model = timm_3d.create_model(
            "resnet34.a1_in1k",#'resnet18.a1_in1k',
            pretrained=True,
            num_classes=2,
            in_chans=1
        )

        self.loss_ce = torch.nn.CrossEntropyLoss(reduction='none')
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y, w = batch
        out = self.model(x)
        
        ce = torch.mean(self.loss_ce(out, y) * w)
        self.log('ce_train', ce, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return ce
        
    def validation_step(self, batch, batch_idx):
        x, y, w = batch
        out = self.model(x)
        
        ce = torch.mean(self.loss_ce(out, y) * w)
        self.log('ce_val', ce, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return ce

    def configure_optimizers(self):
        wd = 0.0
        lr = 2e-5
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        return optimizer

class AugmentedCTDataset(torch.utils.data.Dataset):

    def __init__(self, imgs, labels, weights, mean, std, trial, apply_augmentations=True):
        self.imgs = imgs
        self.labels = labels
        self.weights = weights
        self.mean = mean
        self.std = std
        self.apply_augmentations = apply_augmentations
        self.trial = trial
        self.max_scale = 0.15#self.trial.suggest_float("max_scale", 0, 0.3)
        self.max_translate = 5#self.trial.suggest_float("max_translate", 0, 25)
        self.max_noise_std = 10#self.trial.suggest_float("max_noise_std", 0, 100)
        self.max_deformation_displacement = 3#self.trial.suggest_float("max_deformation_displacement", 0, 5)
        self.augmentations = tio.transforms.Compose([
            tio.transforms.RandomFlip(axes = (0, 1, 2)),
            tio.transforms.RandomElasticDeformation(
                num_control_points=7,
                max_displacement=self.max_deformation_displacement,
                locked_borders=2,
                p=0.05),
            tio.transforms.RandomAffine(
                scales=self.max_scale,
                degrees=180,
                translation=self.max_translate,
                isotropic=False,
                center="image",
                default_pad_value=-1000,
                p=0.9),
            tio.transforms.RandomNoise(mean=0, std=self.max_noise_std, p=0.5)
            ])
        self.counter = 0
        
    def get_item(self, index):
        img = self.imgs[index]
        if self.apply_augmentations: 
            img = self.augmentations(img)
        arr = (img.tensor.to(dtype=torch.float32)-self.mean)/self.std
        return arr, self.labels[index], self.weights[index]
        
    def __getitem__(self, index):
        try:
            return self.get_item(index)
        except Exception as e:
            print(e)

    def __len__(self):
        return len(self.imgs)


def load_scan_and_label(category_folders, name):
    for label, category in enumerate(category_folders):
        for folder in category:
            ct_path = folder / f"{name}.nii.gz"
            if ct_path.exists():
                img = tio.ScalarImage(ct_path)
                return img, label
            

def load_dataset(category_folders, names):
    imgs = []
    labels = []
    weights = []
    for name in names:
        img, label = load_scan_and_label(category_folders, name)
        imgs.append(img)
        labels.append(label)
        weights.append(1.0)
    return imgs, labels, weights

def objective(trial, experiment_folder):
    trial_folder = experiment_folder / "trial_0"
    trial_folder.mkdir()

    base_folder = get_data_folder() / "13.06.2024_cropped_CTs"
    category_folders = [
        [base_folder / "healthy"],
        [base_folder / "injured"]
    ]
    
    aug_val_base_folder = get_data_folder() / "13.06.2024_augmented_val_CTs"
    aug_val_category_folders = [
        [aug_val_base_folder / "healthy"],
        [aug_val_base_folder / "injured"]
    ]
    
    train_names, val_names, _ = get_CT_split_names()
    
    mean = -1000 # not the real mean, but the hounsfield unit of air, to get seamless padding
    std = 188.78704267931099 # the standard deviation was calculated for all voxels > -200 to exclude the background

    train_imgs, train_labels, train_weights = load_dataset(category_folders, train_names)
    val_imgs, val_labels, val_weights = load_dataset(aug_val_category_folders, val_names)

    train_dataset = AugmentedCTDataset(train_imgs, train_labels, train_weights, mean, std, trial)
    val_dataset = AugmentedCTDataset(val_imgs, val_labels, val_weights, mean, std, trial, apply_augmentations=False)
    
    train_loader = utils.data.DataLoader(train_dataset, batch_size=7, num_workers=0, shuffle=True, drop_last=True)
    val_loader = utils.data.DataLoader(val_dataset, batch_size=7, num_workers=0)
    
    l_module = LCTClassificationModule(nr_classes=2, trial=trial)
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=trial_folder / "checkpoints" / "ce_val",
        filename="best_ce_val_{epoch}_{ce_val:.10f}",
        monitor="ce_val",
        save_top_k=1,
        mode="min")

    callbacks = [checkpoint_callback]
    
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=trial_folder / "logs")
    trainer = pl.Trainer(
        max_epochs=10000,
        accelerator="gpu",
        devices=4,
        strategy="ddp_spawn",
        logger=tb_logger,
        log_every_n_steps=1,
        callbacks = callbacks)
    trainer.fit(l_module, train_loader, val_loader)
    
    checkpoint_filename = next((trial_folder / "checkpoints" / "ce_val").glob("best_ce_val_*")).name
    best_model_score = float(checkpoint_filename[checkpoint_filename.find("_ce_val=")+8:-5])
    
    return best_model_score
    

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    experiment_folder = ceu.make_new_experiment_folder(get_results_folder())
    
    objective(0, experiment_folder)
