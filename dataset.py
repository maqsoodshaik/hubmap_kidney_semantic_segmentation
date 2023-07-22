
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
    def __init__(self, json_file: str, data_path: str, class_names: dict = {}):
        import pandas as pd

        # loading json file
        with open(json_file, "r") as json_file:
            json_list = list(json_file)
        # reading label related data
        self.tiles_dicts = []
        for json_str in json_list:
            self.tiles_dicts.append(json.loads(json_str))

        self.datapath = data_path
        self.img_path_template = os.path.join(self.datapath, "{}.tif")

        # getting possible labels
        if class_names is not None:
            self.class_names = class_names
        else:
            print("No class names provided")

    def __len__(self):
        return len(self.tiles_dicts)

    def __getitem__(self, index):
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
            "input": torch.as_tensor(np.array(array), dtype=torch.float32),
            "target": torch.as_tensor(np.array(label_img), dtype=torch.long),
        }
        return torch.as_tensor(np.array(array), dtype=torch.float32).permute(2,1,0), torch.as_tensor(np.array(label_img), dtype=torch.long).permute(1,0)