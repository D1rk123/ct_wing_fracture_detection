from skimage.io import imread
import torch
from torch import optim, utils
import numpy as np
import random
import albumentations as A
import albumentations.pytorch as Ap
import cv2
import pytorch_lightning as pl
from torchvision.models import efficientnet_v2_s

from optuna_pl_callback import PyTorchLightningPruningCallback
from split_names import get_photo_split_names

class LClassificationModule(pl.LightningModule):
    def __init__(self, nr_classes, trial):
        super().__init__()
        self.nr_classes = nr_classes
        self.trial = trial
        self.model = efficientnet_v2_s(weights="IMAGENET1K_V1")

        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = torch.nn.Linear(num_ftrs, self.nr_classes)  

        # Change padding_mode to replicate
        for n, m in self.model.named_modules():
            if isinstance(m, torch.nn.Conv2d):
                if hasattr(m, 'padding_mode'):
                    setattr(m, 'padding_mode', 'replicate')

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
        wd = self.trial.suggest_float("weight_decay", 0.05, 1.0, log=True)
        lr = self.trial.suggest_float("learning_rate", 1e-5, 1e-5*2)
        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)
        return optimizer

class AugmentedPhotoDataset(torch.utils.data.Dataset):

    def __init__(self, imgs, labels, weights, trial, apply_augmentations=True):
        self.imgs = imgs
        self.labels = labels
        self.weights = weights
        self.apply_augmentations = apply_augmentations
        self.result_size = (700, 400)
        self.trial = trial
        self.max_noise_std = self.trial.suggest_float("max_noise_std", 0.05, 0.5, log=True)
        self.max_shear = self.trial.suggest_float("max_shear", 0.0, 15.0)
        self.max_scale = self.trial.suggest_float("max_scale", 1, 1.5)
        self.elastic_alpha_base = self.trial.suggest_float("elastic_alpha_base", 250, 750)
        self.elastic_sigma = self.trial.suggest_float("elastic_sigma", 10, 20)

        if apply_augmentations:
            self.aug_pre = A.Compose([
                A.Flip(p=1),
                A.RandomRotate90(p=1),
                A.OneOf([
                    A.ElasticTransform(
                        alpha=self.elastic_alpha_base,
                        sigma=self.elastic_sigma+self.elastic_alpha_base*0.03,
                        alpha_affine=0,
                        border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
                    A.ElasticTransform(
                        alpha=self.elastic_alpha_base*2,
                        sigma=self.elastic_sigma+self.elastic_alpha_base*0.03,
                        alpha_affine=0,
                        border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
                ], p=0.6667),
                A.OneOf([
                    A.Affine(rotate=(-45, 45), mode=cv2.BORDER_CONSTANT, cval=0, p=0.2),
                    A.Affine(rotate=(-45, 45), scale=(1/self.max_scale, self.max_scale), shear=(-self.max_shear, self.max_shear), mode=cv2.BORDER_CONSTANT, cval=0, p=0.8),
                ], p=1),
                ])
            cjb = self.trial.suggest_float("color_jitter_brightness", 0.001, 0.9, log=True)
            cjc = self.trial.suggest_float("color_jitter_contrast", 0.001, 0.9, log=True)
            cjs = self.trial.suggest_float("color_jitter_saturation", 0.01, 1, log=True)
            cjh = self.trial.suggest_float("color_jitter_hue", 0.00001, 0.9, log=True)
            self.aug_post = A.Compose([
                A.ColorJitter(brightness=cjb, contrast=cjc, saturation=cjs, hue=cjh, p=0.75),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1.0),
                Ap.ToTensorV2()
                ])
        else:
            self.aug_post = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1.0),
                Ap.ToTensorV2()
                ])
                
    def crop_background(self, img):
        # find bounding box
        mask = img[:,:,3] > 0
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        # crop even more if result is too wide
        if cmax-cmin > self.result_size[1]-2:
            crop_width = cmax-cmin - (self.result_size[1]-2)
            cmin += crop_width//2
            cmax = cmin + (self.result_size[1]-2)
        return img[rmin:rmax, cmin:cmax, :]
    
    def add_noise(self, img, s_range, p):
        if np.random.uniform(0, 1) < p:
            img[:,:,0:3] = np.clip(np.random.normal(img[:,:,0:3], np.random.uniform(*s_range)), 0, 1)
        
    def blend(self, img_base, img):
        img_base[:,:,:] = img_base*(1-img[:,:,3,None])
        img_base += img[:,:,0:3]*img[:,:,3,None]
        
    def get_item(self, index):
        img_t = np.pad(self.imgs[index][0], ((100,100),(100,100),(0,0)))
        img_b = np.pad(self.imgs[index][1], ((100,100),(100,100),(0,0)))
        if self.apply_augmentations:
            img_t = self.aug_pre(image=img_t)["image"]
            img_b = self.aug_pre(image=img_b)["image"]
        img_t = self.crop_background(img_t)
        img_b = self.crop_background(img_b)
        if self.apply_augmentations:
            self.add_noise(img_t, (0, self.max_noise_std), 0.75)
            self.add_noise(img_b, (0, self.max_noise_std), 0.75)
        
        combined = np.zeros((self.result_size[0], self.result_size[1], 3), dtype=np.float32)
        # Shuffle blend order (when using data augmentation)
        blends = [(combined[1:img_t.shape[0]+1,1:img_t.shape[1]+1,:], img_t),
            (combined[
            -img_b.shape[0]-1:-1,
            -img_b.shape[1]-1:-1,
            :], img_b)]
        if self.apply_augmentations:
            random.shuffle(blends)
        for blend in blends:
            self.blend(*blend)
        
        return self.aug_post(image=combined)["image"], self.labels[index], self.weights[index]
        
    def __getitem__(self, index):
        try:
            return self.get_item(index)
        except Exception as e:
            print(e)

    def __len__(self):
        return len(self.imgs)
        
def load_img_and_label(category_folders, name):
    for label, category in enumerate(category_folders):
        for folder in category:
            img_path_top = folder / (name + "_top.png")
            img_path_bottom = folder / (name + "_bottom.png")
            if img_path_top.exists() and img_path_bottom.exists():
                img_t = imread(img_path_top).astype(np.float32)/255
                img_b = imread(img_path_bottom).astype(np.float32)/255
                return (img_t, img_b), label
    raise FileNotFoundError(f"Could not find {img_path_top} or {img_path_bottom}")
            

def load_dataset(category_folders, names):
    imgs = []
    labels = []
    weights = []
    for name in names:
        img, label = load_img_and_label(category_folders, name)
        imgs.append(img)
        labels.append(label)
        weights.append(1.0)
    return imgs, labels, weights
    
def objective(trial, experiment_folder, category_folders):
    trial_folder = experiment_folder / f"trial_{trial.number}"
    trial_folder.mkdir()
    
    train_names, val_names, _ = get_photo_split_names()

    train_imgs, train_labels, train_weights = load_dataset(category_folders, train_names)
    val_imgs, val_labels, val_weights = load_dataset(category_folders, val_names)

    train_dataset = AugmentedPhotoDataset(train_imgs, train_labels, train_weights, trial)
    val_dataset = AugmentedPhotoDataset(val_imgs, val_labels, val_weights, trial, apply_augmentations=False)
    
    train_loader = utils.data.DataLoader(train_dataset, batch_size=9, num_workers=0, shuffle=True, drop_last=True)
    val_loader = utils.data.DataLoader(val_dataset, batch_size=9, num_workers=0)
    
    l_module = LClassificationModule(nr_classes=2, trial=trial)
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=trial_folder / "checkpoints" / "ce_val",
        filename="best_ce_val_{epoch}_{ce_val:.10f}",
        monitor="ce_val",
        save_top_k=1,
        mode="min")
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor = "ce_val",
        patience = 100
        )
    optuna_callback = PyTorchLightningPruningCallback(trial, monitor="ce_val")
    callbacks = [checkpoint_callback, early_stopping_callback, optuna_callback]
    
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=trial_folder / "logs")
    trainer = pl.Trainer(
        max_epochs=2000,
        accelerator="gpu",
        devices=4,
        strategy="ddp_spawn",
        logger=tb_logger,
        log_every_n_steps=1,
        callbacks = callbacks)
    trainer.fit(l_module, train_loader, val_loader)
    
    checkpoint_filename = next((trial_folder / "checkpoints" / "ce_val").glob("best_ce_val_*")).name
    best_model_score = float(checkpoint_filename[checkpoint_filename.find("_ce_val=")+8:-5])
    
    optuna_callback.check_pruned()
    
    return best_model_score
