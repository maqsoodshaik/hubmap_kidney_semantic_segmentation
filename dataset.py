
from torch.utils.data import Dataset
import os
import json
import tifffile as tiff
import torch
from PIL import Image
import numpy as np
from utils import get_cartesian_coords
import cv2
class CustomDataset(Dataset):
    '''
    custom dataset creation for hubmap competition
    '''
    def __init__(self, json_file: str = "", data_path: str = "", class_names: dict = {},augment_img: bool = False):
        import pandas as pd
        try:
            # loading json file
            with open(json_file, "r") as json_file:
                json_list = list(json_file)
            # reading label related data
            self.tiles_dicts = []
            for json_str in json_list:
                self.tiles_dicts.append(json.loads(json_str))
            # getting possible labels
            if class_names is not None:
                self.class_names = class_names
            else:
                print("No class names provided")
        except :
            print("No json file provided")
            self.only_image__mode = True
        self.datapath = data_path
        self.img_path_template = os.path.join(self.datapath, "{}.tif")
        self.augment_img = augment_img
        

    def __len__(self):
        return len(self.tiles_dicts)

    def __getitem__(self, index):
        if self.only_image__mode:
            #loop through the images from the data path
            for root, dirs, files in os.walk(self.datapath):
                #if the file is a tif file
                for file in files:
                    if file.endswith(".tif"):
                        #read the image
                        array = tiff.imread(self.img_path_template.format(file.split(".")[0]))
                        if self.augment_img:
                            return {
                            "input": torch.as_tensor(np.array(array), dtype=torch.float32).permute(2,1,0),
                            "target": add_random_occlusions(torch.as_tensor(np.array(array), dtype=torch.float32).permute(2,1,0)),
                             "id": file.split(".")[0]
                            }
                        else:
                            return {
                                "input": torch.as_tensor(np.array(array), dtype=torch.float32).permute(2,1,0),
                                "id": file.split(".")[0]
                            }
                        
        else:
            # reading input image
            id = self.tiles_dicts[index]["id"]
            array = tiff.imread(self.img_path_template.format(id))
            img_example = Image.fromarray(array)
            # creating label image
            annotations = self.tiles_dicts[index]["annotations"]
            # constants
            img_width = img_example.size[0]
            img_height = img_example.size[1]
            label_img = np.zeros(
                (
                    img_width,
                    img_height,
                )
            )
            for annotation in annotations:
                name = annotation["type"]
                indices = get_cartesian_coords(annotation["coordinates"])
                cv2.fillPoly(label_img, [indices], self.class_names[name])
            # creating sample
            sample = {
                "input": torch.as_tensor(np.array(array), dtype=torch.float32).permute(2,1,0),
                "target": torch.as_tensor(np.array(label_img), dtype=torch.long).permute(1,0),
                "id": id
            }
            return sample


def add_random_occlusions(image, max_size=50, num_occlusions=5):
    """
    Add random occlusions to the input image.
    """
    corrupted_image = image.clone()
    for _ in range(num_occlusions):
        x = torch.randint(0, image.shape[-1] - max_size, (1,))
        y = torch.randint(0, image.shape[-2] - max_size, (1,))
        size = torch.randint(10, max_size, (1,))
        
        # Set the pixels of the corrupted_image to 0 in the specified region
        corrupted_image[:, y:y+size, x:x+size] = 0
    
    return corrupted_image

