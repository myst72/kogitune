import os
import json

def load(name, key=None):
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, f'{name}.json')
    with open(file_path) as f:
        data = json.load(f)
        if key:
            return data[key]
        else:
            return data
    