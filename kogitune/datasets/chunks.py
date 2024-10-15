from typing import List
import os
import io
import json
import random
import string
import hashlib
import subprocess
import tempfile
from urllib.parse import urlparse
import numpy as np

from ..loads.commons import *

pyzstd = adhoc.safe_import("pyzstd")

def compress_file(input_file, output_file):
    """
    入力ファイルを読み込み、出力ファイルとして圧縮する
    """
    with open(input_file, "rb") as f_in:
        data = f_in.read()
    compressed_data = pyzstd.compress(data)
    with open(output_file, "wb") as f_out:
        f_out.write(compressed_data)


def decompress_file(input_file, output_file):
    """
    圧縮されたファイルを読み込み、展開して出力ファイルに書き込む
    """
    with open(input_file, "rb") as f_in:
        compressed_data = f_in.read()
    decompressed_data = pyzstd.decompress(compressed_data)
    with open(output_file, "wb") as f_out:
        f_out.write(decompressed_data)


def save_chunk(filepath: str, blocks: List[np.ndarray]):
    if filepath.endswith(".npz"):
        np.savez(filepath, *blocks)


def load_chunk(filepath: str):
    if filepath.endswith(".npz.zst"):
        with open(filepath, "rb") as f:
            byte_data = f.read()
        byte_data = pyzstd.decompress(byte_data)
        byte_stream = io.BytesIO(byte_data)
        npz = np.load(byte_stream, allow_pickle=True)
    else:
        npz = np.load(filepath, allow_pickle=True)
    blocks = [npz[n] for n in npz.files]
    return blocks


def get_filesha1(filepath: str):
    with open(filepath, "rb") as f:
        content = f.read()
        sha1 = hashlib.sha1()
        sha1.update(content)
        sha1_hexdigest = sha1.hexdigest()
    return sha1_hexdigest


def get_filesize(filepath: str):
    if os.path.exists(filepath) and os.path.isfile(filepath):
        return os.path.getsize(filepath)
    else:
        return -1


################


def read_json_from_url(url: str) -> dict:
    requests = adhoc.safe_import("requests")
    response = requests.get(url)

    if response.status_code == 200:
        try:
            # JSONデータをインメモリでパース
            data = response.json()  # json.loads(response.text) でも可
            return data
        except json.JSONDecodeError as e:
            print(repr(e))
    return {}

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
            local_filename = f"_{local_filename}"
        if url.startswith("s3://"):
            download_file_from_s3(url, local_filename)
        else:
            download_file_from_url(url, local_filename)
        return local_filename
    except BaseException as e:
        adhoc.print("ダウンロード失敗", url, repr(e))
        adhoc.exit(throw=e)


def download_file_async(url):
    _, dot, suffix = url.split('/')[-1].partition('.')
    # 空のテンポラリファイルを作成して、ファイルディスクリプタとパスを取得
    fd, temp_file = tempfile.mkstemp(suffix=f'{dot}{suffix}.tmp')
    # すぐにファイルをクローズしてファイルディスクリプタを解放
    os.close(fd)
    # 必要に応じて、後で削除
    # os.unlink(path)
    local_file = temp_file[:-4]
    if url.startswith("s3://"):
        cmd = f"aws s3 cp {url} {temp_file} && mv {temp_file} {local_file}"
    else:
        cmd = f"wget -qO {temp_file} {url} && mv {temp_file} {local_file}"
    print("@", os.path.exists(local_file), cmd)
    try:
        # subprocess.runでコマンドを実行し、エラー時に例外を発生させる
        subprocess.run(f"{cmd} &", shell=True, check=True, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        # エラーをExceptionで捕捉
        print(f"非同期ダウンロード失敗 '{cmd}' failed: {e.stderr.decode()}")
        adhoc.exit(throw=e)
    return local_file


################
#


EMPTY_TOKENS = []


class Packer(adhoc.AdhocObject):
    def __init__(self, kwargs):
        self.pathargs = {}
        self.get(
            kwargs,
            "dataset|=unknown",
            "tokenizer_path|!!",
            "block_size|max_length|!2048",
            "trancate_size|trancate|=0",
            "padding_size|padding|=0",
            "overlap_size|overlap|=0",
        )
        self.tokenizer = adhoc.load("tokenizer", self.tokenizer_path)
        tokens = self.tokenizer.encode("a\nあ\n")
        self.NL_id = tokens[-2]
        self.EOS_id = tokens[-1]
        self.extra_tokens = EMPTY_TOKENS
        self.init_noize(kwargs)

    def new_store(self, base_dir, max_files=512, max_blocks=4096):
        self.store = StoreGenerator(
            self,
            os.path.join(base_dir, self.tokenizer.unique_name()),
            max_files=max_files,
            max_blocks=max_blocks,
        )
        return self.store

    def store_block(self, block: List[int]):
        self.store.write(np.array(block, dtype=np.int32))

    def init_rec(self):  # 新しいレコーダを提供する
        return self.pathargs.copy()

    def encode(self, text: str, rec: dict) -> List[int]:
        extra_tokens = self.extra_tokens
        tokens = self.tokenize(text, rec)
        tokens = self.noize(tokens, rec)
        if len(extra_tokens) == 0:
            rec["num_headblocks"] = rec.get("num_headblocks", 0) + 1
        else:
            tokens = extra_tokens + tokens
        block_size = self.block_size
        start = 0
        while start < len(tokens):
            segmented = tokens[start : start + block_size]
            if len(segmented) == block_size:
                blocked = segmented
            else:
                blocked = self.pad(segmented, rec)
            if len(blocked) == block_size:
                self.store_block(blocked)
                rec["num_blocks"] = rec.get("num_blocks", 0) + 1
                segmented = EMPTY_TOKENS
            start = self.find_next_start(tokens, start + block_size, rec)
        self.extra_tokens = self.trancate_tokens(segmented, rec)

    def tokenize(self, text, rec):
        tokens = self.tokenizer.encode(text)
        rec["num_texts"] = rec.get("num_texts", 0) + 1
        rec["num_chars"] = rec.get("num_chars", 0) + len(text)
        rec["num_tokens"] = rec.get("num_tokens", 0) + (len(tokens) - 1)
        return tokens

    def trancate_tokens(self, extra_tokens, rec):
        if len(extra_tokens) < self.trancate_size:
            rec["num_trancated"] = rec.get("num_trancated", 0) + 1
            rec["num_trancated_tokens"] = rec.get("num_trancated_tokens", 0) + len(
                extra_tokens
            )
            return EMPTY_TOKENS
        return extra_tokens

    def pad(self, tokens, rec):
        length = self.block_size - len(tokens)
        if 0 < length <= self.padding_size:
            # 予想しやすいpaddingを作る
            pad_id = self.EOS_id
            rec["num_padding"] = rec.get("num_padding", 0) + 1
            rec["num_padding_tokens"] = rec.get("num_padding_tokens", 0) + length
            padding = [pad_id] * length
            if length > 2:
                padding = [pad_id] + [self.tokenizer.vocab_size - length] + padding[2:]
            return tokens + padding
        return tokens

    def find_next_start(self, tokens: list, end: int, rec):
        if self.overlap_size > 0:
            # オーバーラップが認められるときは行頭を探す
            tokens = tokens[end - self.overlap_size : end]
            try:
                reverse_index = tokens[::-1].index(self.NL_id)
                rec["num_overlap_tokens"] = (
                    rec.get("num_overlap_tokens", 0) + reverse_index
                )
                rec["num_headblocks"] = rec.get("num_headblocks", 0) + 1
                return end - 1 - reverse_index + 1  # 改行の次なので
            except ValueError as e:
                pass
        if 0 < end < len(tokens) - self.trancate_size and tokens[end - 1] == self.NL_id:
            rec["num_headblocks"] = rec.get("num_headblocks", 0) + 1
        return end

    def init_noize(self, kwargs):
        noize_path = adhoc.get(kwargs, 'noize_map|noize')
        if noize_path:
            self.noize_map = load_token_noize_prob(noize_path)
            self.mask_token_id = adhoc.get(kwargs, 'mask_token_id|mask_id')
            if self.mask_token_id is None:
                mask_token = adhoc.get(kwargs, 'mask_token|mask')
                if mask_token is not None:
                    ids = self.tokenizer.convert_tokens_to_ids([mask_token])
                    adhoc.verbose_print('マスクトークン', mask_token, ids)
                    self.mask_token_id = ids[0]
            self.random_seed = adhoc.get(kwargs, 'random_seed|=42')
        else:
            self.noize_map = None

    def noize(self, tokens:List[int], rec:dict)->List[int]:
        if not self.noize_map:
            return tokens
        random.seed(self.random_seed)
        new_tokens=[tokens[0]]
        if self.mask_token_id is None:
            for t in tokens[1:]:
                if random.random() > self.noize_map[t]:
                    new_tokens.append(t)
            rec['noize_tokens'] = rec.get('noize_tokens', 0) + (len(tokens) - len(new_tokens))
        else:
            masked=0
            for t in tokens[1:]:
                if random.random() > self.noize_map[t]:
                    new_tokens.append(t)
                elif new_tokens[-1] != self.mask_token_id:
                    new_tokens.append(self.mask_token_id)
                    masked+=1
            rec['noize_tokens'] = rec.get('noize_tokens', 0) + (len(tokens) - (len(new_tokens)-masked))
            rec['masked_tokens'] = rec.get('masked_tokens', 0) + masked
        self.random_seed = random.randint(0, 2**31)
        return new_tokens

def load_token_noize_prob(self, noize_path):
    noize_ratio = noize_path if isinstance(noize_path, float) else 0.05
    noize_map = np.full(self.tokenizer.vocab_size, noize_ratio)
    if isinstance(noize_path, str):
        import pandas as pd
        df = pd.read_csv(noize_path)
        for w, r in zip(df['token'], df['ratio']):
            if not isinstance(w, str):
                continue
            ids = self.tokenizer.convert_tokens_to_ids([w])
            noize_map[ids[0]] = r
        adhoc.verbose_print(f'平均ノイズ確率 {noize_map.mean()}', filepath=noize_path)
    noize_map[self.tokenizer.eos_token_id] = 0.0
    return noize_map

################


def generate_random_string(length=8):
    # ランダムな文字列を生成
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))


class StoreGenerator:

    def __init__(self, packer: Packer, base_dir: str, max_files=512, max_blocks=4096):
        self.base_dir = base_dir
        self.packer = packer
        self.max_files = max_files
        self.max_blocks = max_blocks
        self.subdir = None
        self.subdir_files = []
        self.subdir_total = 0
        self.rec = packer.init_rec()
        self.buffers = []

    def encode(self, text: str):
        self.packer.encode(text, self.rec)

    def generate_filepath(self, ext):
        if len(self.subdir_files) > self.max_files:
            self.close_subdir()
        if self.subdir is None:
            while True:
                subdir = os.path.join(self.base_dir, generate_random_string(3))
                if not os.path.exists(subdir):
                    os.makedirs(subdir)
                    break
            self.subdir = subdir

        while True:
            filename = f"{generate_random_string(4)}.{ext}"
            filepath = os.path.join(self.subdir, filename)
            if not os.path.exists(filepath):
                return filepath

    def close_subdir(self):
        if len(self.subdir_files) > 0:
            config = self.rec | {
                "files": self.subdir_files,
                "total_num_tokens": self.subdir_total,
            }
            config_file = os.path.join(self.subdir, "index.json")
            write_config(config_file, config, suffix=None)
            self.subdir = None
            self.subdir_files = []
            self.subdir_total = 0
            self.rec = self.packer.init_rec()

    def save_chunk(self, blocks: List[np.array]):
        filepath = self.generate_filepath("npz")
        save_chunk(filepath, blocks)
        sha1 = get_filesha1(filepath)
        filesize = get_filesize(filepath)
        filepath_zst = f"{filepath}.zst"
        compress_file(filepath, filepath_zst)
        path = filepath_zst[len(self.base_dir) :]
        if path.startswith("/"):
            path = path[1:]
        meta = dict(
            path=path,
            num_blocks=len(blocks),
            block_size=len(blocks[0]),
            filesize=filesize,
            sha1=sha1,
        )
        adhoc.verbose_print("chunk", meta)
        self.subdir_files.append(meta)
        self.subdir_total += len(blocks) * len(blocks[0])

    def write(self, block: np.array):
        self.buffers.append(block)
        while len(self.buffers) >= self.max_blocks:
            blocks = self.buffers[: self.max_blocks]
            self.save_chunk(blocks)
            self.buffers = self.buffers[self.max_blocks :]

    def save(self):
        while len(self.buffers) > 0:
            blocks = self.buffers[: self.max_blocks]
            self.save_chunk(blocks)
            self.buffers = self.buffers[self.max_blocks :]
        self.close_subdir()

    def make_index(self):
        from filelock import FileLock

        config_file = os.path.join(self.base_dir, "index.json")
        lock_file = os.path.join(self.base_dir, "lock")

        with FileLock(lock_file):
            all_files = []
            dedups = {}
            for dir in os.listdir(self.base_dir):
                subdir = os.path.join(self.base_dir, dir)
                index_file = os.path.join(subdir, "index.json")
                if os.path.isdir(subdir) and os.path.exists(index_file):
                    adhoc.print("indexing..", subdir)
                    config = read_config(index_file, suffix=None)
                    files = config.get("files", [])
                    for file in files:
                        sha1 = file["sha1"]
                        if sha1 in dedups:
                            adhoc.print("dedup", file)
                            continue
                        dedups[sha1] = file
                        all_files.append(file)
            config = read_config(config_file, suffix=None)
            config["files"] = all_files
            write_config(config_file, config, suffix=None)


def store(kwargs):
    base_dir = adhoc.get(kwargs, 'store_path|save_path|!!')
    dataset = adhoc.get(kwargs, '_file|dataset|!!')
    data = adhoc.load("datastream", dataset, **kwargs)
    packer = Packer(kwargs)
    store = packer.new_store(base_dir)
    for sample in data.samples():
        text = sample["text"]
        store.encode(text)
    store.save()
    store.make_index()


###


class TokenDataset(adhoc.AdhocObject):

    def __init__(self, base: str, chunkfiles: List[dict], **kwargs):
        super().__init__(base, kwargs)
        total = 0
        files = []
        self.block_size = self.get(kwargs, "block_size|max_length|=512")
        for meta in chunkfiles:
            chunk_block_size = meta["block_size"]
            num_blocks = meta["num_blocks"]
            if self.block_size <= chunk_block_size:
                n_factor = chunk_block_size // self.block_size
                meta["num_blocks"] = num_blocks * n_factor
                total += meta["num_blocks"]
                files.append(meta)
                path = meta["path"]
                filepath = os.path.join(base, path)
                if os.path.isfile(filepath):
                    meta["filepath"] = filepath
                # print("@meta", meta)
        # TODO: rank support
        # if get_world_size() > 1:
        #     rank = get_rank()
        #     world_size = get_world_size()
        #     self.chunk_files = [f for i, f in enumerate(self.chunk_files) if i % world_size == rank]
        #     self.n_items = len(self.chunk_files) * self.n_blocks
        #     logs.append(f'ランク rank={rank}/{world_size}')
        self.pathargs = {}  # ここから記憶する
        self.random_seed = self.get(kwargs, "random_seed|=0")
        if self.random_seed != 0:
            random.seed(self.random_seed)
            random.shuffle(files)
        self.total = total
        self.files = files
        self.resize(*self.get(kwargs, "start|=0", "end", "length|size"))
        self.reindex(self.get(kwargs, "resume_index|=0"))
        self.loaded_chunk_index = -1
        self.try_prefetch(self.chunk_index)

    def index(self, files, index):
        base_index = 0
        for chunk_index, meta in enumerate(files):
            offest = meta.get("offset", 0)
            chunk_size = meta["num_blocks"] - offest
            if index < base_index + chunk_size:
                inner_index = index - base_index
                return chunk_index, inner_index
            base_index += chunk_size
        return chunk_index, chunk_size  # こんなことはありえない

    def resize(self, start, end, size):
        start = self.calc_index(start)
        end = self.calc_index(end)
        size = self.calc_index(size)
        if isinstance(size, int):
            end = start + size
        if not isinstance(end, int):
            end = self.total
        if end < start:
            end += self.total
        n_times = end // self.total
        if end % self.total > 0:
            n_times += 1
        # print(f"@start={start} @end={end} @n_times={n_times}")
        files = []
        for _ in range(n_times):
            files.extend(self.files)
        start_index, start_offset = self.index(files, start)
        end_index, end_end = self.index(files, end)
        files = files[start_index : end_index + 1]
        for i in range(len(files)):
            files[i] = files[i].copy()
        files[0]["offset"] = start_offset
        files[-1]["num_blocks"] = end_end
        # print(f"@total {self.total} => {end - start}")
        self.total = end - start
        self.files = files

    def calc_index(self, n):
        if isinstance(n, str):
            if n[-1] == "M" or n[-1] == "m":
                n = (1000_000 * float(n[:-1])) / (self.total * self.block_size)
            elif n[-1] == "B" or n[-1] == "b":
                n = (1000_000_000 * float(n[:-1])) / (self.total * self.block_size)
            elif n[-1] == "T" or n[-1] == "t":
                n = (1000_000_000_000 * float(n[:-1])) / (self.total * self.block_size)
        if isinstance(n, str) and n.endswith("%"):
            n = float(n[:-1]) / 100
        if isinstance(n, float) and n < 10:
            return int(self.total * n)
        return n

    def reindex(self, start):
        chunk_index, inner_index = self.index(self.files, start % self.total)
        self.full_index = start
        self.chunk_index = chunk_index
        self.chunk_size = len(self.files)
        self.inner_index = inner_index + self.files[chunk_index].get("offest", 0)
        self.inner_size = self.files[chunk_index]["num_blocks"]

    def __len__(self):
        return self.total

    def __getitem__(self, index):
        block = self.get_block(self.chunk_index, self.inner_index)
        self.inner_index += 1
        self.full_index += 1
        if self.inner_index < self.inner_size:
            return block
        self.chunk_index = (self.chunk_index + 1) % len(self.files)
        self.inner_index = self.files[self.chunk_index].get("offest", 0)
        self.inner_size = self.files[self.chunk_index]["num_blocks"]
        self.try_prefetch(self.chunk_index)
        return block

    def get_block(self, chunk_index, inner_index):
        if self.loaded_chunk_index != chunk_index:
            blocks = self.load_blocks(chunk_index)
            self.blocks = self.resize_blocks(blocks)
            self.loaded_chunk_index = chunk_index
        return self.blocks[inner_index]

    def resize_blocks(self, blocks):
        newblocks = []
        for array in blocks:
            # 配列の長さを block_size で割り切れる長さに調整
            trimmed_length = (len(array) // self.block_size) * self.block_size
            array = array[:trimmed_length]
            num_splits = trimmed_length // self.block_size
            splits = np.array_split(array, num_splits)
            newblocks.extend(splits)
        return newblocks

    def load_blocks(self, chunk_index):
        meta = self.files[chunk_index]
        if "filepath" in meta:
            filepath = meta["filepath"]
            if meta.get("is_tempfile", False):
                os.unlink(filepath)
                del meta["filepath"]
                del meta["is_tempfile"]
            if get_filesize(filepath) > 0:
                blocks = load_chunk(filepath)
                return blocks
        url = os.path.join(self.path, meta["path"])
        adhoc.verbose_print('非同期ダウンロード失敗', url)
        print("@meta", chunk_index, meta, url)
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".npz.zst", delete=True) as temp_file:
            download_file(url, temp_file.name)
            blocks = load_chunk(temp_file.name)
            return blocks

    def try_prefetch(self, chunk_index):
        meta = self.files[chunk_index]
        if "filepath" in meta and os.path.exists(meta["filepath"]):
            return True
        url = os.path.join(self.path, meta["path"])
        meta["filepath"] = download_file_async(url)
        meta["is_tempfile"] = True


class StoreLoader(adhoc.AdhocLoader):

    def load_from_map(self, path, kwargs):
        if ".json" in path:
            config = read_config(path, suffix=None)
            try:
                base_url, files = config["base_url"], config["files"]
            except KeyError as e:
                adhoc.print(f"{path}の形式が違うね", adhoc.dump(config))
                raise e
            return TokenDataset(base_url, files, **kwargs)
        if path.startswith("https://"):
            base, files = find_index_from_url(path, **kwargs)
            return TokenDataset(base, files, **kwargs)
        if os.path.isdir(path):
            base, files = find_index_from_local(path, **kwargs)
            return TokenDataset(base, files, **kwargs)


StoreLoader({}).register("chunk")

UNAME = {}


def get_unique_name(tokenizer_path):
    if tokenizer_path not in UNAME:
        tokenizer = adhoc.load("tokenizer", tokenizer_path)
        UNAME[tokenizer_path] = tokenizer.unique_name()
    return UNAME[tokenizer_path]


def find_index_from_local(base_dir, **kwargs):
    config = read_config(os.path.join(base_dir, "index.json"), suffix=None)
    if "files" in config:
        return base_dir, config["files"]
    tokenizer_path = adhoc.get(kwargs, "tokenizer_path|tokenizer|!!")
    unique_name = get_unique_name(tokenizer_path)
    extended_base_dir = os.path.join(base_dir, unique_name)
    config = read_config(os.path.join(extended_base_dir, "index.json"), suffix=None)
    if "files" in config:
        return extended_base_dir, config["files"]
    return FileNotFoundError(f"No index.json in {base_dir}")


def find_index_from_url(base_url, **kwargs):
    config = read_json_from_url(os.path.join(base_url, "index.json"))
    if "files" in config:
        return base_url, config["files"]
    tokenizer_path = adhoc.get(kwargs, "tokenizer_path|tokenizer")
    if tokenizer_path:
        unique_name = get_unique_name(tokenizer_path)
        extended_base_url = os.path.join(base_url, unique_name)
        config = read_json_from_url(os.path.join(extended_base_url, "index.json"))
        if "files" in config:
            return extended_base_url, config["files"]
    # 旧Kogituneへの対応
    for block_size in [512, 1024, 256, 2048, 4096]:
        config = read_json_from_url(
            os.path.join(base_url, f"text{block_size}train_config.json")
        )
        if "files" in config:
            files = []
            total = config["n_items"]
            num_blocks = config["n_chunks"]
            for path, meta in config["files"].items():
                meta["path"] = f"{path}.zst"
                meta["block_size"] = config["max_length"]
                meta["num_blocks"] = total if total < num_blocks else num_blocks
                total -= meta["num_blocks"]
                files.append(meta)
            assert total == 0
            return base_url, files

    return FileNotFoundError(f"No index.json in {base_url}")


# def test_store(base_dir, block_size=512):
#     load.adhoc('store': "/")
#     config_file = os.path.join(base_dir, "index.json")
#     config = read_config(config_file)
#     print(adhoc.dump(config))
#     dataset = TokenDataset(base_dir, block_size, config["files"])
#     for i in range(len(dataset)):
#         dataset[i]


# # https://papertown-jwu.s3.ap-northeast-1.amazonaws.com/llm-jp-915a/mc4ja_line/0135/text512train_00.npz.zst


# def makeindex(base_url, prefix="text512train_"):
#     config_file = f"{basename(base_url)}.json"
#     config = {"base_url": base_url}
#     files = []
#     for i in range(2):
#         for j in range(2):
#             url = f"{base_url}/{i:04d}/{prefix}{j:02d}.npz.zst"
#             print(url)
#             data = check(url)
#             if data is None:
#                 break
#             files.append(data)
#         if j == 0:
#             break
#     config["files"] = files
#     write_config(config_file, config)
