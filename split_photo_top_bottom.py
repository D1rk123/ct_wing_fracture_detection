import skimage.io
from skimage.measure import label, regionprops
import numpy as np

from folder_locations import get_data_folder

if __name__ == "__main__":
    base_path = get_data_folder() / "19.01.2024 WING FRACTURES 4 CATEGORIES" / "PHOTOGRAPHS" 
    category_paths = [
        base_path / "CATEGORY 1 - NO BRUISE + NO FRACTURE (old HEAL)",
        base_path / "CATEGORY 2 - BRUISE + NO FRACTURE (SOFT TISS)",
        base_path / "CATEGORY 3 - NO BRUISE + FRACTURE (old POST)",
        base_path / "CATEGORY 4 - BRUISE + FRACTURE (old PRE)"
        ]
    out_base_folder = get_data_folder() / "19.01.2024 WING FRACTURES 4 CATEGORIES" / "PHOTOGRAPHS_SPLIT"
    
    widths = []
    heights = []
    for folder in category_paths:
        out_folder = (out_base_folder / folder.name)
        out_folder.mkdir(exist_ok=True, parents=True)
        for img_path in folder.glob("*.png"):
            img = skimage.io.imread(img_path)
            img_mask = img[:,:,3] > 0
            img_labels = label(img_mask)
            img_regions = regionprops(img_labels)
            if len(img_regions) < 2:
                print(f"Too few regions: {img_path}")
            img_regions = sorted(img_regions, key=lambda x:x.num_pixels, reverse=True)
            img_regions = sorted(img_regions[0:2], key=lambda x:x.centroid[0])
            for region, name in zip(img_regions, ["top", "bottom"]):
                min_row, min_col, max_row, max_col = region.bbox
                sub_img_mask = img_labels == region.label
                sub_img = (img*sub_img_mask[:,:,None])[min_row:max_row, min_col:max_col, :]
                save_path = out_folder / (img_path.name[:-4]+f"_{name}.png")
                skimage.io.imsave(save_path, sub_img)
                widths.append(sub_img.shape[1])
                heights.append(sub_img.shape[0])
    print(f"max_width = {np.max(widths)}, max_height = {np.max(heights)}")
            
    
