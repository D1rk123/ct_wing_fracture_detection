import torch
import torch.nn.functional as F
from torcheval.metrics import BinaryAUROC
import numpy as np

from folder_locations import get_results_folder, get_data_folder
from nn_ct import load_dataset, LCTClassificationModule, AugmentedCTDataset
from split_names import get_CT_split_names

if __name__ == "__main__":
    base_path = get_data_folder() / "13.06.2024_cropped_CTs"
    category_folders = [
        [base_path / "healthy"],
        [base_path / "injured"]
    ]
    
    checkpoint_path = get_results_folder() / "2024-07-18_nn_ct_1" / "trial_0" / "checkpoints" / "ce_val" / "best_ce_val_epoch=9558_ce_val=0.0002484065.ckpt"
    
    l_module = LCTClassificationModule.load_from_checkpoint(checkpoint_path)
    l_module.freeze()
    l_module.eval()
    
    train_names, val_names, test_names = get_CT_split_names()
    
    mean = -1000 # not the real mean, but the Hounsfield unit of air, to get seamless padding
    std = 188.78704267931099 # the standard deviation was calculated for all voxels > -200 to exclude the background

    selected_names = test_names
    imgs, labels, weights = load_dataset(category_folders, selected_names)

    dataset = AugmentedCTDataset(imgs, labels, weights, mean, std, 0, apply_augmentations=False)
    
    errors = []
    class1_softmaxes = []
    confusion_matrix = np.zeros((2, 2), dtype=int)
    
    print("label,prediction,confidence,name")
    
    for i, name in zip(range(len(dataset)), selected_names):
        preds = []
        confs = []
        
        img, label, weight = dataset[i]
        out = l_module(img[None,:,:,:].cuda()).cpu()[0]
        softmax = F.softmax(out)
        pred = torch.argmax(softmax)
        conf = torch.max(softmax)
        
        round_pred = np.round(pred.numpy())
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
