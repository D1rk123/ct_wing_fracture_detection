import torch
import torch.nn.functional as F
from torcheval.metrics import BinaryAUROC
import numpy as np

from folder_locations import get_results_folder, get_data_folder
from nn_photo_utils import load_dataset, AugmentedPhotoDataset, LClassificationModule
from split_names import get_photo_split_names

if __name__ == "__main__":
    base_path = get_data_folder() / "13.06.2024_NEW_PHOTOGRAPHS_WINGS_FRACTURE" / "PHOTOGRAPHS_SPLIT"   
    
    goal = "f" # "f" = fracture, "b" = bruise
    if goal == "f":
        category_folders = [
            [base_path / "CATEGORY 1 - NO BRUISE + NO FRACTURE (old HEAL)", base_path / "CATEGORY 2 - BRUISE + NO FRACTURE (SOFT TISS)"],
            [base_path / "CATEGORY 3 - NO BRUISE + FRACTURE (old POST)", base_path / "CATEGORY 4 - BRUISE + FRACTURE (old PRE)"]
        ]
        checkpoint_path = get_results_folder() / "2024-08-15_nn_photo_fractures_1" / "trial_122" / "checkpoints" / "ce_val" / "best_ce_val_epoch=489_ce_val=0.0184715797.ckpt"
    else:
        category_folders = [
            [base_path / "CATEGORY 1 - NO BRUISE + NO FRACTURE (old HEAL)", base_path / "CATEGORY 3 - NO BRUISE + FRACTURE (old POST)"],
            [base_path / "CATEGORY 2 - BRUISE + NO FRACTURE (SOFT TISS)", base_path / "CATEGORY 4 - BRUISE + FRACTURE (old PRE)"]
        ]
        checkpoint_path = get_results_folder() / "2024-08-15_nn_photo_bruises_2" / "trial_223" / "checkpoints" / "ce_val" / "best_ce_val_epoch=334_ce_val=0.0784088315.ckpt"
    
    l_module = LClassificationModule.load_from_checkpoint(checkpoint_path)
    l_module.freeze()
    l_module.eval()
    
    trial = l_module.trial

    train_names, val_names, test_names = get_photo_split_names()

    selected_names = test_names
    imgs, labels, weights = load_dataset(category_folders, selected_names)

    dataset = AugmentedPhotoDataset(imgs, labels, weights, trial, apply_augmentations=False)
    
    errors = []
    class1_softmaxes = []
    confusion_matrix = np.zeros((2, 2), dtype=int)
    
    print("label,prediction,confidence,name")
    
    for i, name in zip(range(len(dataset)), selected_names):
        preds = []
        confs = []
        
        img, label, weight = dataset[i]
        out = l_module(img[None,:,:,:].cuda()).cpu()[0]
        softmax = F.softmax(out, dim=0)
        pred = torch.argmax(softmax).item()
        conf = torch.max(softmax)
        
        round_pred = np.round(pred)
        print(f"{label},{round_pred},{conf},{name}")
        errors.append((label != round_pred).astype(np.float64).item())
        confusion_matrix[label, round_pred.astype(int)] += 1
        class1_softmaxes.append(softmax[1].numpy().item())
        
    print("Label order: healthy, injured")
    print(f"Accuracy = {(1-np.mean(errors))*100}%")
    print(confusion_matrix)
    
    auc_metric = BinaryAUROC()
    auc_metric.update(torch.tensor(class1_softmaxes, dtype=torch.float64), torch.tensor(labels, dtype=torch.float64))
    auc = auc_metric.compute()
    print(f"AUC = {auc}")
