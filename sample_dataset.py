import pandas as pd
import random
import json
import os

from .data_utils import load_dataframe

SCHEMA_DIR = "../schema"

def sample_datasets(paths=None, types=["DocumentSimilarity"]):
    if paths is None:
        _, _, files = os.walk(SCHEMA_DIR)
        paths = [os.path.join(SCHEMA_DIR, file) for file in files]

    for path in paths:
        with open(path, "r") as f:
            task = json.load(f)
            data = load_dataframe(task["train_file"]["file_path"], task["train_file"]["file_type"], task["train_file"]["file_header"])
            print(data)


sample_datasets()