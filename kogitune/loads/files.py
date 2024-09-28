from typing import List
import os
import json
from urllib.parse import urlparse

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


def download_file_from_url(url, local_filename):
    requests = adhoc.safe_import("requests")

    with requests.get(url, stream=True) as r:
        r.raise_for_status()  # ダウンロードのステータスを確認
        # ダウンロードしたファイルをローカルに保存
        with open(local_filename, "wb") as f:
            for chunk in r.iter_content(
                chunk_size=8192
            ):  # ファイルを少しずつダウンロード
                f.write(chunk)


def download_file_from_s3(s3_url, local_filename):
    """
    # 使用例
    s3_url = "s3://my-bucket/path/to/file.txt"
    download_file_from_s3(s3_url, "downloaded_file.txt")
    """
    boto3 = adhoc.safe_import("boto3")

    # S3 URLを解析してバケット名とオブジェクトキーを取得
    parsed_url = urlparse(s3_url)

    if parsed_url.scheme != "s3":
        raise ValueError(f"Invalid S3 URL: {s3_url}")

    bucket_name = parsed_url.netloc  # バケット名
    s3_key = parsed_url.path.lstrip("/")  # ファイルのパス

    # S3クライアントの作成
    s3 = boto3.client("s3")

    # ファイルをダウンロード
    s3.download_file(bucket_name, s3_key, local_filename)


def download_file(url, local_filename=None):
    try:
        if local_filename is None:
            local_filename = url.rpartition("/")[-1]
        if url.startswith("s3://"):
            download_file_from_s3(url, local_filename)
        else:
            download_file_from_url(url, local_filename)
        return True
    except BaseException as e:
        adhoc.print("@download", repr(e))
        return False
