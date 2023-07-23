import numpy as np 
import json

def get_cartesian_coords(coords):
    coords_array = np.array(coords).squeeze()

    return coords_array




def read_secret_from_json(json_file, secret_name):
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    # Check if the secret name exists in the JSON data
    if secret_name in data:
        return data[secret_name]
    else:
        raise KeyError(f"Secret '{secret_name}' not found in the JSON file.")

   