import sys
import torch
import torch.multiprocessing
import optuna
import functools

import ct_experiment_utils as ceu
from folder_locations import get_results_folder, get_data_folder
from nn_photo_utils import objective


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    experiment_folder = ceu.make_new_experiment_folder(get_results_folder())
    
    base_path = get_data_folder() / "13.06.2024_NEW_PHOTOGRAPHS_WINGS_FRACTURE" / "PHOTOGRAPHS_SPLIT"
    category_folders = [
        [base_path / "CATEGORY 1 - NO BRUISE + NO FRACTURE (old HEAL)", base_path / "CATEGORY 3 - NO BRUISE + FRACTURE (old POST)"],
        [base_path / "CATEGORY 2 - BRUISE + NO FRACTURE (SOFT TISS)", base_path / "CATEGORY 4 - BRUISE + FRACTURE (old PRE)"]
    ]
    
    study_name = "chicken_wing_photo_bruises"
    
    # The storage is provided as the first command line argument
    # We used a mysql server for these experiments

    study = optuna.create_study(
        study_name=study_name,
        storage=sys.argv[1],
        direction="minimize",
        pruner=optuna.pruners.NopPruner(),
        load_if_exists=True,
    )
    
    study.optimize(
        functools.partial(objective, experiment_folder=experiment_folder, category_folders=category_folders),
        n_trials=250,
        timeout=None,
        n_jobs=1
    )
