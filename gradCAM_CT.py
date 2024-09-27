import torch
import torch.nn.functional as F
import numpy as np
from captum.attr import LayerGradCam, GuidedGradCam
import torchio as tio

import ct_experiment_utils as ceu
from folder_locations import get_results_folder, get_data_folder
from nn_ct import load_dataset, LCTClassificationModule, AugmentedCTDataset
from split_names import get_CT_split_names


if __name__ == "__main__":
    base_path = get_data_folder() / "13.06.2024_cropped_CTs"
    experiment_folder = ceu.make_new_experiment_folder(get_results_folder())
    category_folders = [
        [base_path / "healthy"],
        [base_path / "injured"]
    ]
    
    aug_val_base_folder = get_data_folder() / "13.06.2024_augmented_val_CTs"
    aug_val_category_folders = [
        [aug_val_base_folder / "healthy"],
        [aug_val_base_folder / "injured"]
    ]
    
    checkpoint_path = get_results_folder() / "2024-07-18_nn_ct_1" / "trial_0" / "checkpoints" / "ce_val" / "best_ce_val_epoch=9558_ce_val=0.0002484065.ckpt"
    
    l_module = LCTClassificationModule.load_from_checkpoint(checkpoint_path)
    l_module.freeze()
    l_module.eval()
    
    gradCAM = LayerGradCam(l_module, l_module.model.layer4)
    guidedGradCAM = GuidedGradCam(l_module, l_module.model.layer4)

    train_names, val_names, test_names = get_CT_split_names()
    
    selected_names = test_names
    
    mean = -1000 # not the real mean, but the hounsfield unit of air, to get seamless padding
    std = 188.78704267931099 # the standard deviation was calculated for all voxels > -200 to exclude the background

    imgs, labels, weights = load_dataset(category_folders, selected_names)

    dataset = AugmentedCTDataset(imgs, labels, weights, mean, std, 0, apply_augmentations=False)
    
    errors = []
    class1_softmaxes = []
    confusion_matrix = np.zeros((2, 2), dtype=int)
    
    for i, name in zip(range(len(dataset)), selected_names):
    #for i, name in zip(range(3), selected_names):
        img, label, weight = dataset[i]
        out = l_module(img[None,:,:,:,:].cuda()).cpu()[0]
        softmax = F.softmax(out, dim=0)
        pred = torch.argmax(softmax).item()
        
        folder_name = ""
        
        if label == 0:
            folder_name += "label_healthy"
        else:
            folder_name += "label_fractured"
        
        if pred < 0.5:
            folder_name += "_predicted_healthy"
        else:
            folder_name += "_predicted_fractured"
            
        save_folder = experiment_folder / folder_name
        if not save_folder.exists():
            save_folder.mkdir()
        
        
        gc = gradCAM.attribute(img[None, ...].cuda(), target=1)
        gc_upsampled = gradCAM.interpolate(gc, (196, 196, 196)).detach().cpu().numpy()[0, 0, :, :, :]
        gc = gc.detach().cpu().numpy()[0, 0, :, :, :]
        ggc = guidedGradCAM.attribute(img[None, ...].cuda(), target=1).detach().cpu().numpy()[0, 0, :, :, :]
        
        transf_lo_res = tio.transforms.Resize(gc.shape)
        
        tio_img = imgs[i]
        tio_img.save(save_folder/ f"{i}.nii")
        
        tio_gc = transf_lo_res(tio_img)
        tio_gc.set_data(gc[None, ...])
        tio_gc.save(save_folder/ f"{i}_GradCAM.nii")
        
        tio_img.set_data(gc_upsampled[None, ...])
        tio_img.save(save_folder/ f"{i}_GradCAM_upsampled.nii")
        
        tio_img.set_data(ggc[None, ...])
        tio_img.save(save_folder/ f"{i}_guided_GradCAM.nii")
        
