from typing import List, Union, Any
import os
import json
from urllib.parse import urlparse, parse_qs, urlencode

from .prints import aargs_print, use_ja


def edit_distance(s1, s2):
    if len(s1) < len(s2):
        return edit_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def find_simkey(dic, given_key, max_distance=1):
    key_map = {}
    for key in dic.keys():
        if key not in key_map:
            key_map[key] = edit_distance(key, given_key)
    keys = sorted([(dis, k) for k, dis in key_map.items() if dis <= max_distance])
    if len(keys) > 0:
        aargs_print(keys, verbose="simkey")
        return keys[0][1]
    return None


## コンフィグファイル


def load_yaml(config_file):
    import yaml

    loaded_data = {}
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
        for section, settings in config.items():
            if isinstance(settings, dict):
                for key, value in settings.items():
                    loaded_data[key] = value
        return loaded_data


def load_json(config_file):
    with open(config_file, "r") as file:
        return json.load(file)


def load_list(list_file, convert_fn=str):
    with open(list_file) as f:
        return [
            convert_fn(line.strip())
            for line in f.readlines()
            if line.strip() != "" and not line.startswith("#")
        ]


def load_config(config_file, default_value={}):
    if isinstance(config_file, dict):
        return config_file
    if config_file.endswith(".json"):
        return load_json(config_file)
    if config_file.endswith(".yaml"):
        return load_yaml(config_file)
    if config_file.endswith("_list.txt"):
        return load_list(config_file)
    return default_value


def parse_key_value(key: str, value: Union[str, int]):
    if not isinstance(value, str):
        return value
    try:
        return int(value)
    except:
        pass
    try:
        return float(value)
    except:
        pass
    lvalue = value.lower()
    if lvalue == "true":
        return True
    if lvalue == "false":
        return False
    if key.startswith("_list") and value.endsith(".txt"):
        return load_list(value)
    if key.startswith("_config") and value.endsith("_args"):
        return load_config(value, default_value=value)
    if key.startswith("_comma") and value.endsith("_camma"):
        return value.split(",")
    return value


def list_keys(keys: Union[List[str], str], sep="|"):
    if isinstance(keys, (list, tuple)):
        return keys
    if sep != "|":
        return [k.strip() for k in keys.split(sep)]
    return keys.split(sep)


def list_values(values: Any, map_fn=str):
    if isinstance(values, list):
        return values
    if isinstance(values, (tuple, set)):
        return list(values)
    if isinstance(values, dict):
        return list(values.keys())
    if values is None:
        return []
    if isinstance(values, str):
        if "|" in values:
            return [map_fn(x) for x in values.split("|")]
        return [map_fn(x) for x in values.split(",")]
    return [values]


def normal_key(key):
    if key.endswith("_comma") or key.endswith("_camma"):
        return key[:-6]
    return key


def find_dict_from_keys(dic: dict, keys: List[str], default_value=None):
    keys = list_keys(keys)
    default_key = normal_key(keys[0])
    for key in keys:
        if key.startswith("="):
            value = parse_key_value(default_key, key[1:])
            if dic.get('verbose', True) or dic.get('use_panda', False):
                aargs_print(f"{default_key}={repr(value)}", face='🐼', once=True)
            return default_key, value
        if key.startswith("!!"):
            raise KeyError(f"'{default_key}'.")
        if key.startswith("!"):
            value = parse_key_value(default_key, key[1:])
            aargs_print(
                f"`{default_key}` is missing. Confirm the default"
                f"//`{default_key}`が設定されてないよ. 確認して",
                f"{default_key}={repr(value)}."
            )
            return default_key, value
        if key in dic:
            return key, dic.get(key)
        use_simkey = key.count("*")
        if use_simkey > 0:
            key = key.replace("*", "")
            simkey = find_simkey(dic, key, max_distance=use_simkey)
            if simkey in dic:
                return key, dic.get(simkey)
    return default_key, default_value


def copy_dict_from_keys(src_args: dict, dist_args: dict, *keys_list: List[str]):
    for keys in keys_list:
        keys = list_keys(keys)
        default_key = keys[0]
        for key in keys:
            if key in src_args:
                dist_args[default_key] = src_args[key]
                break


def move_dict_from_keys(src_args: dict, dist_args: dict, *keys_list: List[str]):
    for keys in keys_list:
        keys = list_keys(keys)
        default_key = keys[0]
        for key in keys:
            if key in src_args:
                dist_args[default_key] = src_args.pop(key)
                break


def extract_dict_with_keys(dic: dict, *keys):
    extracted = {}
    for key in dic.keys():
        if key in keys:
            extracted[key] = dic[key]
    return extracted


def extract_dict_with_prefix(dic: dict, prefix: str):
    extracted = {}
    if not prefix.endswith("_"):
        prefix = f"{prefix}_"
    for key in dic.keys():
        if key.startswith(prefix):
            subkey = key[len(prefix) :]
            if subkey in ("config", "kwargs", "args", "options"):
                extracted.update(load_config(dic[key]))
            elif subkey == "path":
                pass
            else:
                extracted[subkey] = dic[key]
    return extracted


def safe_kwargs(kwargs: dict, prefix=None, /, *keys_list):
    extracted = {}
    for keys in keys_list:
        keys = list_keys(keys)
        for key in keys:
            if key in kwargs:
                extracted[keys[0]] = kwargs[key]
                break
    if prefix:
        kwargs_sub = extract_dict_with_prefix(kwargs, prefix)
        extracted = extracted | kwargs_sub
    return extracted


def parse_path(path: str, parent_args={}):
    if 0 < path.find("#") < path.find("?"):
        path, _, query = path.partition("?")
        if query != "":
            query = f"?{query}"
        path, _, fragment = path.partition("#")
        if fragment != "":
            fragment = f"#{fragment}"
        path = f"{path}{query}{fragment}"

    parsed_url = urlparse(path)
    options = parse_qs(parsed_url.query)
    args = {k: parse_key_value(k, v[0]) for k, v in options.items()}
    if len(parsed_url.scheme):
        if parsed_url.port:
            path = f"{parsed_url.scheme}://{parsed_url.netloc}:{parsed_url.port}{parsed_url.path}"
        elif parsed_url.netloc != "":
            path = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
        else:
            path = f"{parsed_url.scheme}:{parsed_url.path}"
    else:
        path = f"{parsed_url.path}"
    return path, parent_args | args, parsed_url.fragment


def encode_path(path: str, tag: str, kwargs: dict):
    query = "" if kwargs is None or len(kwargs) == 0 else f"?{urlencode(kwargs)}"
    fragment = "" if tag == "" else f"#{tag}"
    return f"{path}{query}{fragment}"


def parse_path_args(path: str, parent_args=None, include_urlinfo=False):
    """
    pathから引数を読み込む
    """
    if path.startswith("{") and path.startswith("}"):
        ## JSON形式であれば、最初のパラメータはパス名、残りは引数として解釈する。
        args = json.loads(path)
        first_key = list(args.keys())[0]
        if parent_args is None:
            return args.pop(first_key), args
        return args.pop(first_key), ChainMap(args, parent_args)

    parsed_url = urlparse(path)
    options = parse_qs(parsed_url.query)
    args = {k: parse_key_value(k, v[0]) for k, v in options.items()}
    if len(parsed_url.scheme):
        if parsed_url.port:
            url = f"{parsed_url.scheme}://{parsed_url.netloc}:{parsed_url.port}{parsed_url.path}#{parsed_url.fragment}"
        elif parsed_url.netloc != "":
            url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}#{parsed_url.fragment}"
        else:
            url = f"{parsed_url.scheme}:{parsed_url.path}#{parsed_url.fragment}"
    else:
        url = f"{parsed_url.path}#{parsed_url.fragment}"
    if include_urlinfo:
        args["url_scheme"] = parsed_url.scheme
        args["url_host"] = parsed_url.netloc
        if parsed_url.username:
            args["userame"] = parsed_url.username
            args["password"] = parsed_url.password
        if parsed_url.port:
            args["port"] = parsed_url.port
        args["path"] = parsed_url.path
    if parent_args is None:
        return url, args
    return url, parent_args | args


def use_os_environ():
    env = {}
    for key in os.environ.keys():
        if key.islower():
            env[key] = parse_key_value(key, os.environ[key])
    return env


class ChainMap(object):
    def __init__(self, dic: dict, parent: dict = None, read_only=False):
        self.parent = parent
        self.local_dic = {} if dic is None else dic
        self.used_keys = []
        self.read_only = read_only  # この層の更新は行わない

    def __repr__(self):
        if self.parent is None:
            return repr(self.local_dic)
        return f"{self.local_dic} {self.parent}"

    def __contains__(self, key):
        if key in self.local_dic:
            return True
        if self.parent is None:
            return False
        return key in self.parent

    def get(self, key, default_value=None):
        if "|" in key or "*" in key:
            return find_dict_from_keys(self, list_keys(key))[1]
        if key in self.local_dic:
            self.use_key(key)
            return self.local_dic[key]
        if self.parent is None:
            return default_value
        return self.parent.get(key, default_value)

    def pop(self, key, default_value=None):
        if key in self.local_dic:
            self.use_key(key)
            if self.parent is not None:
                self.parent.pop(key, None)  # 捨てる
            return self.local_dic.pop(key, default_value)
        if self.parent is None:
            return default_value
        return self.parent.pop(key, default_value)

    def keys(self):
        keys = list(self.local_dic.keys())
        if self.parent is not None:
            for key in self.parent.keys():
                if key not in keys:
                    keys.append(key)
        return keys

    def use_key(self, key):
        self.used_keys.append(key)
        if hasattr(self.parent, "use_key"):
            self.parent.use_key(key)

    def unused_keys(self):
        unused_keys = []
        for key in self.local_dic.keys():
            if key not in self.used_keys:
                unused_keys.append(key)
        return unused_keys

    def __getitem__(self, key):
        if "|" in key or "*" in key:
            return find_dict_from_keys(self, list_keys(key))[1]
        if key in self.local_dic:
            self.use_key(key)
            return self.local_dic[key]
        if self.parent is None:
            return None
        return self.parent.get(key, None)

    def __setitem__(self, key, value):
        if self.read_only:
            # もし読み込み専用なら、親を更新する
            if self.parent is not None:
                self.parent[key] = value
        else:
            self.local_dic[key] = value
            self.use_key(key)

    def record(self, *keys, field=None, dic=None):
        """ """
        for key in keys:
            key, value = find_dict_from_keys(self, key)
            if isinstance(dic, dict):
                if not key.startswith("_"):
                    dic[key] = value
            if field is not None:
                if key.startswith("_"):
                    setattr(field, key[1:], value)
                else:
                    setattr(field, key, value)


## データ操作
import re


def extract_keys_from_format(s):
    # 正規表現を使って{}で囲まれた部分を全て抽出
    matches = re.findall(r"\{([^}]+)\}", s)
    # フォーマット指定を考慮して、コロン以前の部分（キー名）のみを取り出す
    keys = [match.split(":")[0] for match in matches]
    # 重複を除いたリストを返す
    return list(set(keys))


def safe_format(format: str, **kwargs):
    try:
        return format.format(**kwargs)
    except KeyError:
        keys = extract_keys_from_format(format)
        for key in keys:
            if key not in kwargs:
                kwargs[key] = f"({key})"
        return format.format(**kwargs)


def get_formatted_text(sample: dict, key_or_format: str):
    if "{" in key_or_format:
        return safe_format(key_or_format, **sample)
    return sample[key_or_format]


