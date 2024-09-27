import torch
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from skimage.transform import resize
from captum.attr import LayerGradCam

import ct_experiment_utils as ceu
from folder_locations import get_results_folder, get_data_folder
from nn_photo_utils import load_dataset, LClassificationModule, AugmentedPhotoDataset
from split_names import get_photo_split_names

if __name__ == "__main__":
    base_path = get_data_folder() / "13.06.2024_NEW_PHOTOGRAPHS_WINGS_FRACTURE" / "PHOTOGRAPHS_SPLIT"
    
    goal = "b" # "f" = fracture, "b" = bruise
    if goal == "f":
        category_folders = [
            [base_path / "CATEGORY 1 - NO BRUISE + NO FRACTURE (old HEAL)", base_path / "CATEGORY 2 - BRUISE + NO FRACTURE (SOFT TISS)"],
            [base_path / "CATEGORY 3 - NO BRUISE + FRACTURE (old POST)", base_path / "CATEGORY 4 - BRUISE + FRACTURE (old PRE)"]
        ]
        checkpoint_path = get_results_folder() / "2024-08-15_nn_photo_fractures_1" / "trial_122" / "checkpoints" / "ce_val" / "best_ce_val_epoch=489_ce_val=0.0184715797.ckpt"
        label_names = {0 : "Not fractured", 1: "Fractured"}
        injury_name = "fractured"
    else:
        category_folders = [
            [base_path / "CATEGORY 1 - NO BRUISE + NO FRACTURE (old HEAL)", base_path / "CATEGORY 3 - NO BRUISE + FRACTURE (old POST)"],
            [base_path / "CATEGORY 2 - BRUISE + NO FRACTURE (SOFT TISS)", base_path / "CATEGORY 4 - BRUISE + FRACTURE (old PRE)"]
        ]
        checkpoint_path = get_results_folder() / "2024-08-15_nn_photo_bruises_2" / "trial_223" / "checkpoints" / "ce_val" / "best_ce_val_epoch=334_ce_val=0.0784088315.ckpt"
        label_names = {0 : "Not bruised", 1: "Bruised"}
        injury_name = "bruised"
    
    l_module = LClassificationModule.load_from_checkpoint(checkpoint_path)
    l_module.freeze()
    l_module.eval()
    
    trial = l_module.trial

    train_names, val_names, test_names = get_photo_split_names()
    
    selected_names = test_names

    imgs, labels, weights = load_dataset(category_folders, selected_names)

    dataset = AugmentedPhotoDataset(imgs, labels, weights, trial, apply_augmentations=False)
    
    experiment_folder = ceu.make_new_experiment_folder(get_results_folder())
    figures_folder = experiment_folder / "figures"
    figures_folder.mkdir()
    
    gradCAM = LayerGradCam(l_module, l_module.model.features[7])
    
    mean=torch.tensor((0.485, 0.456, 0.406))
    std=torch.tensor((0.229, 0.224, 0.225))
    
    for i, name in tqdm(zip(range(len(dataset)), selected_names)):        
        img, label, weight = dataset[i]
        out = l_module(img[None,:,:,:].cuda()).cpu()[0]
        softmax = F.softmax(out, dim=0)
        pred = torch.argmax(softmax).item()
        
        heatmap = gradCAM.attribute(img[None, ...].cuda(), target=1).detach().cpu().numpy()[0, 0, :, :]
        
        img = np.moveaxis(torch.clip(img * std[:, None, None] + mean[:, None, None], 0, 1).cpu().numpy(), 0, 2)
        
        vmax = 0.3
        vmin = -vmax
        
        # Get the color map by name:
        cm = plt.get_cmap('PRGn_r')

        # Apply the colormap like a function to any array:
        cm_heatmap = cm(np.interp(heatmap,[vmin,vmax],[0,1]))
        
        plt.figure(figsize=(12, 5))
        plt.subplot(131)
        plt.imshow(img)
        plt.title(f"label = {label_names[label]}, nn_out = {label_names[pred]}")
        plt.subplot(132)
        plt.imshow(heatmap, vmin=vmin, vmax=vmax, cmap = cm)
        plt.colorbar()
        plt.title(f"GradCAM on {injury_name} label")
        plt.subplot(133)
        plt.imshow(0.4*img+0.6*resize(cm_heatmap[:,:,:3], img.shape))
        plt.title("Merged")
        plt.tight_layout()
        plt.savefig(figures_folder / f"{i}.png")
        plt.close()
