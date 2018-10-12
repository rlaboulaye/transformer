import json
import os

import pandas as pd
import numpy as np

from data.data_utils import load_dataframe

SCHEMA_DIR = "schema/"

def sample_dataset(task_path, name=None, num_datasets=None, max_length=300):
    with open(task_path, "r") as f:

        if name is None:
            head, tail = os.path.split(task_path)
            name = tail.replace("_task.json", "").replace(".json", "") + "_sampled"
        task_dir = os.path.join(head, name)
        if not os.path.exists(task_dir):
            os.mkdir(task_dir)

        task = json.load(f)
        data_head, data_tail = os.path.split(task["train_file"]["file_path"])
        data_path = os.path.join(data_head, name + "_datasets")
        if not os.path.exists(data_path):
            os.mkdir(data_path)

        data = load_dataframe(
            task["train_file"]["file_path"],
            task["train_file"]["file_type"],
            task["train_file"]["file_header"])
        data = data.sample(frac=1)
        target = data.columns[task["target"]["column_indices"]][0]

        num_datasets = data.shape[0] // max_length
        sub_datasets = data.groupby(target)
        splits = []
        for _, dataset in sub_datasets:
            splits.append(np.array_split(dataset, num_datasets))
        subsets = []
        for x in range(num_datasets):
            subset = pd.DataFrame()
            for split in splits:
                subset = subset.append(split.pop(0))
            subsets.append(subset.sample(frac=1))

        for i, dataframe in enumerate(subsets, 1):
            # file_name = name + "_" + str(i) + "_of_" + str(num_datasets) + ".csv"
            file_name = f"{name}_{i}_of_{num_datasets}.csv"
            dataframe.to_csv(os.path.join(data_path, file_name))
            sub_task = task.copy()
            sub_task["train_file"]["file_path"] = data_path
            sub_task["train_file"]["file_type"] = "csv"
            sub_task["target"]["num_classes"] = dataframe[target].nunique()
            with open(os.path.join(task_dir, file_name), "w") as fh:
                json.dump(sub_task, fh, indent=4)


sample_dataset("./schema/airline_task.json")