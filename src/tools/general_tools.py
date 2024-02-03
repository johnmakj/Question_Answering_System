import json
import os
import pickle
from datetime import timedelta
from functools import wraps
from time import time
from typing import Dict

import numpy as np
import torch
import yaml


def get_folder_path(path_from_module: str) -> str:
    """Method to find the folders that in many cases is needed but are not visible.
    Args:
        path_from_module (str): the path from the central repo to the folder

    Returns:
        str
    """
    fn = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__))).split('src')[0]
    return '{0}{1}'.format(fn, path_from_module)


def get_filepath(path_from_module: str, file_name: str) -> str:
    """Method to find the path-files that in many cases is needed but are not visible.
    Args:
        path_from_module (str): the path from the central repo to the folder
        file_name (str): the file we want from the folder

    Returns:
        str, the actual path to folder
    """
    fn = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__))).split('src')[0]
    return '{0}{1}/{2}'.format(fn, path_from_module, file_name)


def time_it(method):  # pragma: no cover
    """Print the runtime of the decorated method"""

    # Required for time_it to also work with other decorators.
    @wraps(method)
    def timed(*args, **kwargs):
        start = time()
        result = method(*args, **kwargs)
        finish = time()

        print(f'Execution completed in {timedelta(seconds=round(finish - start))} s. '
              f'[method: <{method.__name__}>]')
        return result

    return timed


def load_pickled_data(path: str):  # pragma: no cover
    """Load data from pickle

    Args:
        path (str): path to file

    Returns:
        Pickled Object: data
    """
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)

        return data

    except FileNotFoundError:
        raise FileNotFoundError(f'File/Dir {path} is not found.')


def dump_pickled_data(path: str, data: object) -> None:  # pragma: no cover
    """Write data to pickle

    Args:
        path (str): path to file
        data (object): data
    """
    try:
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    except FileNotFoundError:
        raise FileNotFoundError(f'File/Dir {path} is not found.')


def load_json_file(filepath: str) -> Dict:
    """Reads the specified JSON file.

    Args:
        filepath (str): The JSON filepath to read.

    Returns:
        Dict
    """
    try:
        with open(filepath, 'r') as json_file:
            return json.load(json_file)

    except FileNotFoundError:
        raise FileNotFoundError(f'JSON file{filepath} is not found')


def load_yaml_config(filepath: str) -> Dict:
    """
    A method to load a yaml config file
    Args:
        filepath (str): The path of the file (including the name of the config file)

    Returns:
        dict
    """
    try:
        with open(filepath) as file:
            config = yaml.safe_load(file)

        return config

    except FileNotFoundError:
        raise FileNotFoundError(f'Yaml file{filepath} is not found')


GLOBAL_SEED = 1


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


GLOBAL_WORKER_ID = None


def _init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)
