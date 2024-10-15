from typing import List
import os
import json

from .commons import adhoc


def zopen(filepath, mode="rt"):
    if filepath.endswith(".zst"):
        pyzstd = adhoc.safe_import('pyzstd')
        return pyzstd.open(filepath, mode)
    elif filepath.endswith(".gz"):
        import gzip
        return gzip.open(filepath, mode)
    else:
        return open(filepath, mode)


def basename(path: str, split_ext=True, split_dir=True):
    if "?" in path:
        path = path.partition("?")[0]
    if "#" in path:
        path = path.partition("#")[0]
    if split_dir and "/" in path:
        path = path.rpartition("/")[-1]
    if split_dir and "\\" in path:
        path = path.rpartition("\\")[-1]
    if split_ext and "." in path:
        if path.endswith('.jsonl'):
            return path[:-6]
        path = path.partition(".")[0]
    return path


def get_extention(path: str):
    base = basename(path, split_dir=False)
    return path[len(base) :]


def join_name(path, subname=None, sep='_', ext=None):
    if subname is not None:
        path = f"{path}{sep}{subname}"
    if ext is not None:
        path = f"{path}.{ext}"
        if '..' in path:
            path = path.replace('..', '.')
    return path

def safe_makedirs(save_path):
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory) and directory != "":
        os.makedirs(directory)


def get_config_file(filepath: str, suffix="_config"):
    if suffix is None or filepath.endswith(f"{suffix}.json"):
        return filepath
    name = basename(filepath, split_dir=False)
    return f"{name}{suffix}.json"


def read_config(filepath: str, suffix="_config"):
    config_file = get_config_file(filepath, suffix=suffix)
    if os.path.exists(config_file):
        with open(config_file) as f:
            return json.load(f)
    return {}


def write_config(filepath: str, config: dict, suffix="_config"):
    config_file = get_config_file(filepath, suffix=suffix)
    with open(config_file, "w") as w:
        return json.dump(config, w, ensure_ascii=False, indent=2)


def get_num_of_lines(filepath):
    config = read_config(filepath)
    if "num_of_lines" in config:
        return config["num_of_lines"]
    with zopen(filepath) as f:
        c = 0
        line = f.readline()
        while line:
            c += 1
            line = f.readline()
    config["num_of_lines"] = c
    write_config(filepath, config)
    return c


def file_jsonl_reader(filepath: str, start=0, end=None):
    if end is None:
        end = get_num_of_lines(filepath)
    pbar = adhoc.progress_bar(total=end - start, desc=filepath)
    # pbar = tqdm(total=end - start, desc=filepath)
    with zopen(filepath) as f:
        line = f.readline()
        c = start
        while line and c < end:
            pbar.update()
            yield json.loads(line) if line.startswith("{") else {"text": line.strip()}
            line = f.readline()
            c += 1
        pbar.close()


