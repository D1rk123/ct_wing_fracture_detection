#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 14:55:50 2023

@author: des
"""
import itertools
import random

from folder_locations import get_data_folder

def split_train_val_test(li, val_frac, test_frac):
    li_len = len(li)
    val_li = random.sample(li, round(val_frac*li_len))
    li = [el for el in li if el not in val_li]
    test_li = random.sample(li, round(test_frac*li_len))
    train_li = [el for el in li if el not in test_li]
    
    return train_li, val_li, test_li
    
def categorize_dataset(category_paths):
    dataset = []
    for category_path in category_paths:
        for image_path in category_path.glob("*.png"):
            dataset.append((category_path, image_path.name[0], image_path))
    return dataset
    

if __name__ == "__main__":
    base_path = get_data_folder() / "19.01.2024 WING FRACTURES 4 CATEGORIES" / "PHOTOGRAPHS"
    category_paths = [
        base_path / "CATEGORY 1 - NO BRUISE + NO FRACTURE (old HEAL)",
        base_path / "CATEGORY 2 - BRUISE + NO FRACTURE (SOFT TISS)",
        base_path / "CATEGORY 3 - NO BRUISE + FRACTURE (old POST)",
        base_path / "CATEGORY 4 - BRUISE + FRACTURE (old PRE)"
        ]
        
    dataset = categorize_dataset(category_paths)

    train_dataset = []
    val_dataset = []
    test_dataset = []
    for day_letter, category_path in itertools.product(["a", "b", "c", "d", "e", "f", "g"], category_paths):
        selection = [sample for sample in dataset if sample[0] == category_path and sample[1] == day_letter]
        
        train_selection, val_selection, test_selection = split_train_val_test(selection, 0.2, 0.2)
        train_dataset += train_selection
        val_dataset += val_selection
        test_dataset += test_selection

    for dataset in [train_dataset, val_dataset, test_dataset]:
        dataset.sort(key = lambda sample : sample[2].name)
    
    print(f"train ({len(train_dataset)}): {[sample[2].name[:-4] for sample in train_dataset]}")
    print(f"val ({len(val_dataset)}): {[sample[2].name[:-4] for sample in val_dataset]}")
    print(f"test ({len(test_dataset)}): {[sample[2].name[:-4] for sample in test_dataset]}")
