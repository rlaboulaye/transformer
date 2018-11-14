import json
import os

import pandas as pd
import numpy as np

from data.data_utils import load_dataframe

SCHEMA_DIR = "schema/"


def sample_dataset(task_path, name=None, num_datasets=None, max_length=300):

    # Get task json object
    with open(task_path, "r") as f:
        task = json.load(f)

    # load the dataframe
    data = load_dataframe(
        task["train_file"]["file_path"],
        task["train_file"]["file_type"],
        task["train_file"]["file_header"])
    data = data.sample(frac=1)
    target = data.columns[task["target"]["column_indices"]][0]

    # Split the dataset without overlap, maintaining class ratios
    if num_datasets is None:
        subsets = split_dataset(data, target, length=max_length)
        num_datasets = len(subsets)
    else:
        subsets = split_dataset(data, target, num_datasets=num_datasets)

    # If name is not supplied, create a name of the form '{task title}_sampled'
    if name is None:
        task_head, task_tail = os.path.split(task_path)
        name = task_tail.replace("_task.json", "").replace(".json", "") + "_sampled"

    # Make a directory for the new task json files inside the current task directory
    task_dir = os.path.join(task_head, name + "_tasks")
    if not os.path.exists(task_dir):
        os.mkdir(task_dir)

    # Make a directory for the new sampled datasets inside the current dataset directory
    data_head, data_tail = os.path.split(task["train_file"]["file_path"])
    data_dir = os.path.join(data_head, name + "_datasets")
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    for i, dataframe in enumerate(subsets, 1):
        # Save the sampled dataset in data_dir
        data_file_name = f"{name}_{i}_of_{num_datasets}.csv"
        data_path = os.path.join(data_dir, data_file_name)
        dataframe.to_csv(data_path, index=False)

        # Create a new task object and save it in task_dir
        task_file_name = f"{name}_task_{i}_of_{num_datasets}.json"
        task_path = os.path.join(task_dir, task_file_name)
        sub_task = task.copy()
        sub_task["train_file"]["file_path"] = data_path
        sub_task["train_file"]["file_type"] = "csv"  # all new datasets are csv's
        sub_task["target"]["num_classes"] = dataframe[
            target].nunique()  # update in case of a very small class not being included
        with open(task_path, "w") as fh:
            json.dump(sub_task, fh, indent=4)


def split_dataset(dataframe, target_col, length=None, num_datasets=None):
    """Splits a dataframe into subsets without any overlap, maintaining class ratios"""

    if not ((length is not None and num_datasets is None) or
            (length is None and num_datasets is not None)):
        raise ValueError("Please specify either length or num_datasets but not both")
    if num_datasets is not None and num_datasets > dataframe.shape[0]:
        raise ValueError("num_datasets must be smaller than dataframe")

    # Determine the new number of datasets from length if not given
    if num_datasets is None:
        num_datasets = dataframe.shape[0] // length

    # Each class is split into num_datasets pieces
    splits = []
    for _, dataset in dataframe.groupby(target_col):
        splits.append(np.array_split(dataset, num_datasets))

    # One chunk from each class is put back into a subset and shuffled
    subsets = []
    for x in range(num_datasets):
        subset = pd.DataFrame()
        for split in splits:
            subset = subset.append(split.pop(0))
        subsets.append(subset.sample(frac=1))

    return subsets


# sample_dataset("/users/guest/m/masonfp/Desktop/transformer/airline_twitter_sentiment/airline_task.json")
