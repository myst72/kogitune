from typing import Optional, List, Union, Any
import os
import sys
import time
import re
import json
import inspect
from urllib.parse import urlparse, parse_qs, urlencode

try:
    from termcolor import colored
except ModuleNotFoundError:
    def colored(text, color):
        return text

## Stacked Arguments

ARGS_STACK = []

class ChainMap(object):
    def __init__(self, kwargs: dict, parent = None, 
                 caller = None, errors='ignore', open_file = None):
        self.local_kwargs = {} if kwargs is None else kwargs
        self.parent = parent
        self.caller = caller
        self.errors = errors
        self.used_keys = []
        self.saved_files = []
        if open_file is not None:
            self.file = open(open_file, "w")
            self.opened = True
        else:
            self.file = parent.file if parent is not None else None
            self.opened = False

    def __repr__(self):
        if self.parent is None:
            return repr(self.local_kwargs)
        return f"{self.local_kwargs} {self.parent}"

    def __contains__(self, key):
        if key in self.local_kwargs:
            return True
        if self.parent is None:
            return False
        return key in self.parent

    def get(self, key, default_value=None, use_key=False):
        cur = self
        while cur is not None:
            if key in cur.local_kwargs:
                if use_key:
                    cur.used_keys.append(key)
                return cur.local_kwargs[key]
            cur = cur.parent
        return default_value

    def pop(self, key, default_value=None):
        if key in self.local_kwargs:
            # self.use_key(key)
            if self.parent is not None:
                self.parent.pop(key, None)  # 捨てる
            return self.local_kwargs.pop(key, default_value)
        if self.parent is None:
            return default_value
        return self.parent.pop(key, default_value)

    def keys(self):
        keys = list(self.local_kwargs.keys())
        if self.parent is not None:
            for key in self.parent.keys():
                if key not in keys:
                    keys.append(key)
        return keys

    def __getitem__(self, key):
        cur = self
        while cur is not None:
            if key in cur.local_kwargs:
                # self.use_key(key) # これを追加すると、**kwargs でusedになる。
                return cur.local_kwargs[key]
            cur = cur.parent
        raise KeyError(key)

    def __setitem__(self, key, value):
        self.local_kwargs[key] = value
        self.use_key(key)

    def get_kwargs(self):
        if self.parent is None:
            return self.kwargs
        else:
            return self.parent.get_kwargs() | self.kwargs

    def __enter__(self):
        global ARGS_STACK
        if len(ARGS_STACK) > 0:
            self.parent = ARGS_STACK[-1] 
        ARGS_STACK.append(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        global ARGS_STACK
        if exc_type is None:
            self.report_saved_files()
            self.report_unused_keys()
        if self.opened and self.file:
            self.file.close()
            self.file = None
        ARGS_STACK.pop()

    def use_key(self, key):
        if key and not key.startswith('_'):
            cur = self
            while cur is not None:
                if key in cur.local_kwargs:
                    cur.used_keys.append(key)
                cur = cur.parent

    def unused_keys(self):
        unused_keys = []
        for key in self.local_kwargs.keys():
            if key.startswith('_') or key in self.used_keys:
                continue
            if key in self.parent:
                continue
            unused_keys.append(key)
        return unused_keys

    def report_unused_keys(self): 
        unused_keys = self.unused_keys()
        if len(unused_keys) > 0:
            # if self.errors == "ignore":
            #     return
            if self.errors == "strict":
                raise TypeError(f"{key} is an unused keyword at {self.caller}")
            adhoc_print(f"List of unused arguments//未使用の引数があるよ！スペルミスない？", self.caller)
            for key in unused_keys:
                value = self[key]
                adhoc_print(f"  {key} = {repr(value)}", face='')

    def saved(self, filepath: str, desc:str='', rename_from=None):
        if rename_from and os.path.exists(rename_from):
            if os.path.exists(filepath):
                os.remove(filepath)
            os.rename(rename_from, filepath)
        if filepath not in [file for file, _ in self.saved_files]:
            self.saved_files.append((filepath, desc))

    def report_saved_files(self):
        if len(self.saved_files) == 0:
            return
        width = max(len(filepath) for filepath, _ in self.saved_files) + 8
        for filepath, desc in self.saved_files:
            adhoc_print(colored(filepath.ljust(width), "blue"), desc)

## stacked

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

def parse_key_value(key: str, value: str):
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
    if lvalue == "none" or lvalue == "null":
        return None
    if key.startswith("_config") and value.endsith("_args"):
        return load_config(value, default_value=value)
    return value

def get_os_environ():
    env = {}
    for key in os.environ.keys():
        if key.islower():
            env[key] = parse_key_value(key, os.environ[key])
    return env

ARGS_STACK.append(ChainMap(get_os_environ()))

def get_stacked(key:Optional[str]=None, default_value=None):
    stack = ARGS_STACK[-1]
    if key is None:
        return stack
    return stack.get(key, default_value, use_key=True)


def use_stacked_key(key:Optional[str]):
    if key is not None:
        ARGS_STACK[-1].use_key(key)

def saved_on_stacked(filepath: str, desc:str='', rename_from=None):
    ARGS_STACK[-1].saved(filepath, desc, rename_from=rename_from)


def kwargs_from_stacked(caller_frame=None, errors = 'ignore', open_file=None, /, **kwargs) -> ChainMap:
    stacked = get_stacked()
    if caller_frame is None:
        caller_frame = inspect.stack()[1].function
    kwargs.pop('open_file', None) # 余分なキーを消す
    return ChainMap(kwargs, parent=stacked, caller=caller_frame, errors=errors)


### print

def _split_en(text, sep='///'):
    if isinstance(text, str):
        return text.split(sep)[0]
    return str(text)

def _split_ja(text, sep='///'):
    if isinstance(text, str):
        return text.split(sep)[-1]
    return str(text)

def dict_as_json(data: dict):
    if isinstance(data, (int, float, str, bool)) or data is None:
        return data
    elif isinstance(data, dict):
        d = {}
        for key, value in data.items():
            try:
                d[key] = dict_as_json(value)
            except ValueError as e:
                pass
        return d
    elif isinstance(data, (list, tuple)):
        return [dict_as_json(x) for x in data]
    else:
        raise ValueError()

def dump_dict_as_json(data: dict, indent=2):
    return json.dumps(dict_as_json(data), indent=indent, ensure_ascii=False)

def stringfy_kwargs(**kwargs):
    ss = []
    for key, value in kwargs.items():
        key = colored(key, 'blue')
        ss.append(f"{key}={repr(value)}")
    return ss

ONCE = {}

def init_once():
    global ONCE
    ONCE = {}

def adhoc_print(*args, **kwargs):
    global ONCE, ARGS_STACK
    face = kwargs.pop("face", get_stacked("face", "🦊"))
    once = kwargs.pop("once", False)
    color = kwargs.pop("color", None)
    sep = kwargs.pop("sep", " ")
    end = kwargs.pop("end", os.linesep)
    text_en = sep.join(_split_en(a) for a in args)
    text_ja = sep.join(_split_ja(a) for a in args)
    if once:
        once_key = once if isinstance(once, str) else text_en
        if once_key in ONCE:
            return
        ONCE[once_key] = True
    if color:
        text_en = colored(text_en, color)
        text_ja = colored(text_ja, color)
    print(f"{face}{text_en}", end=end)
    if text_en != text_ja and end == os.linesep:
        print(f"{face}{text_ja}", end=end)
    file = ARGS_STACK[-1].file
    if file:
        print(f"{face}{text_en}", end=end, file=file)


def is_verbose():
    return get_stacked('verbose', 5) > 0


def verbose_print(*args, **kwargs):
    if is_verbose():
        if 'color' not in kwargs:
            kwargs['color'] = 'cyan'
        adhoc_print(*args, **kwargs)

def is_debug():
    return get_stacked('debug', True)


def debug_print(*args, **kwargs):
    if is_debug():
        DEBUG = colored(f"DEBUG[{inspect.stack()[1].function}]", "red")
        adhoc_print(DEBUG, *args, **kwargs)

def notice(*args, **kwargs):
    adhoc_print(*args, *stringfy_kwargs(**kwargs))
    sys.stdout.flush()

def warn(*args, **kwargs):
    adhoc_print(colored("FIXME" "red"), *args, *stringfy_kwargs(**kwargs))

def notice_kwargs(path, kwargs, exception=None):
    if isinstance(exception, BaseException):
        adhoc_print(repr(exception))
    adhoc_print(path, kwargs)

def function_called(name, *args, **kwargs):
    strargs = []
    for a in args:
        strargs.append(repr(a))
    for key, value in kwargs.items():
        strargs.append(f'{key}={repr(value)}')
    strargs = ','.join(strargs)
    return f"{name}({strargs})"

def report_ArgumentError(message, throw, called):
#    adhoc_print(f"An error has occured.///エラーが発生しました", repr(throw))
    adhoc_print(message, repr(throw))
    adhoc_print("Your parameter could be wrong.///あなたのパラメータ指定が間違っている可能性が高いです。", repr(throw))
    adhoc_print(called)
    if is_debug():
        raise throw
    adhoc_print("If you want to know the detail///エラーの詳細を知りたい場合は", "`debug=True`")
    sys.exit(1)

## Adhoc Keys

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
        debug_print(keys)
        return keys[0][1]
    return None

def identity(x: Any):
    return x

def rec_value(dic: dict, value, recursive=1, type_fn = identity):
    if isinstance(value, str) and recursive > 0 and '{' in value:
        keys = extract_keys_from_format(value)
        if len(keys) > 0:
            values = [get_adhoc(dic, key, recursive-1) for key in keys]
            for key, val in zip(keys, values):
                value = value.replace(f'{{{key}}}', f'{val}')
        return identity(value)
    return identity(value)

def get_adhoc(dic: dict, 
                       adhoc_key: str, 
                       default_value=None, 
                       recursive=3, 
                       return_keys = True, 
                       type_fn=identity):
    keys = list_keys(adhoc_key)
    default_key = keys[0]
    for key in keys:
        matched_key = None
        if key.startswith("!!"):
            print('@', adhoc_key)
            raise KeyError(default_key)
        if key.startswith("="):
            value = parse_key_value(default_key, key[1:])
            if dic.get('verbose', True) or dic.get('use_panda', False):
                adhoc_print(f"{default_key}={repr(value)}", face=' 🐼', once=True)
        elif key.startswith("!"):
            value = parse_key_value(default_key, key[1:])
            adhoc_print(
                f"`{default_key}` is missing. Confirm the default"
                f"///`{default_key}`が設定されてないよ. 確認して",
                f"{default_key}={repr(value)}."
            )
        elif key in dic:
            matched_key = key
            use_stacked_key(key)
            value = dic.get(key)
        else:
            use_simkey = key.count("*")
            if use_simkey == 0:
                continue
            key = key.replace("*", "")
            simkey = find_simkey(dic, key, max_distance=use_simkey)
            if simkey not in dic:
                continue
            matched_key = simkey
            use_stacked_key(matched_key)
            value = dic.get(simkey)
        value = rec_value(dic, value, recursive, type_fn)
        return (value, matched_key, default_key) if return_keys else value
    return (default_value, None, default_key) if return_keys else default_value


def record(kwargs: dict, *adhoc_keys: str, record_to=None, record_dic=None):
    values = []
    for adhoc_key in adhoc_keys:
        recordable = True
        if adhoc_key.startswith("_"):  # _で始まるときは残さない
            recordable = False
            adhoc_key = adhoc_key[1:]
        value, matched_key, default_key = get_adhoc(kwargs, adhoc_key)
        values.append(value)
        if recordable and isinstance(record_dic, dict) and matched_key is not None:
            record_dic[default_key] = value
        if record_to is not None:
            if default_key.endswith('_path') and hasattr(record_to, "subpath") and record_to.subpath == "":
                default_key = "subpath"
            setattr(record_to, default_key, value)
    return values[0] if len(values) == 1 else values

def get(kwargs: dict, adhoc_key: str, default_value=None, 
        use_global=False, return_keys=False, type_fn=identity):
    if use_global:
        kwargs = get_stacked().get_kwargs() | kwargs
    return get_adhoc(kwargs, adhoc_key, default_value, return_keys=return_keys)

def get_list(kwargs: dict, keys: str, use_global=False):
    value = get(kwargs, keys, use_global=use_global, return_keys=False)
    return list_values(value)

# def copy_dict_from_keys(src_args: dict, dist_args: dict, *keys_list: List[str]):
#     for keys in keys_list:
#         keys = list_keys(keys)
#         default_key = keys[0]
#         for key in keys:
#             if key in src_args:
#                 dist_args[default_key] = src_args[key]
#                 break


# def move_dict_from_keys(src_args: dict, dist_args: dict, *keys_list: List[str]):
#     for keys in keys_list:
#         keys = list_keys(keys)
#         default_key = keys[0]
#         for key in keys:
#             if key in src_args:
#                 dist_args[default_key] = src_args.pop(key)
#                 break

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


def safe_kwargs(kwargs: dict, prefix=None, /, *adhoc_keys):
    extracted = {}
    for keys in adhoc_keys:
        keys = list_keys(keys)
        for key in keys:
            if key in kwargs:
                extracted[keys[0]] = kwargs[key]
                break
    if prefix:
        kwargs_sub = extract_dict_with_prefix(kwargs, prefix)
        extracted = extracted | kwargs_sub
    for key in extracted.keys():
        use_stacked_key(key)
    return extracted


## Adhoc formatting 

def extract_keys_from_format(s:str) -> list[str]:
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


### main arguments

_key_pattern = re.compile(r"^[A-Za-z0-9\.\-_]+\=")

def _parse_argv_key_value(key, next_value, args):
    if _key_pattern.match(key):
        if key.startswith("--"):
            key = key[2:]
        key, _, value = key.partition("=")
        return key, parse_key_value(key, value)
    elif key.startswith("--"):
        key = key[2:]
        if next_value.startswith("--"):
            if key.startswith("enable_") or key.startswith("enable-"):
                return key[7:], True
            elif key.startswith("disable_") or key.startswith("disable-"):
                return key[8:], False
            return key, True
        else:
            args["_"] = next_value
            return key, parse_key_value(key, next_value)
    else:
        if args.get("_") != key:
            files = args.get("files", [])
            files.append(key)
            args["files"] = files
    return key, None


def parse_argv(argv: List[str], expand_config="config"):
    # argv = sys.argv[1:]
    args = {"_": ""}
    for arg, next_value in zip(argv, argv[1:] + ["--"]):
        key, value = _parse_argv_key_value(arg, next_value, args)
        if value is not None:
            key = key.replace("-", "_")
            if key == expand_config:
                loaded_data = load_config(value)
                args.update(loaded_data)
            else:
                args[key.replace("-", "_")] = value
    del args["_"]
    return args


def kwargs_from_main(use_subcommand=False, expand_config="config", /, **kwargs):
    import sys
    if use_subcommand and len(sys.argv) > 1:
        args = parse_argv(sys.argv[2:], expand_config=expand_config)
        args["subcommand"] = sys.argv[1]
    else:
        args = parse_argv(sys.argv[1:], expand_config=expand_config)
    return ChainMap(args, get_stacked(), caller='main')

## kwargs

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


def kwargs_from_path(path: str, errors='ignore', /, **kwargs):
    _path, kwargs, _tag = parse_path(path, kwargs)
    kwargs = kwargs_from_stacked(path, errors, **kwargs)
    kwargs['_path'] = _path
    kwargs['_tag'] = _tag
    return kwargs

def encode_path(path: str, tag: str, kwargs: dict):
    query = "" if kwargs is None or len(kwargs) == 0 else f"?{urlencode(kwargs)}"
    fragment = "" if tag == "" else f"#{tag}"
    return f"{path}{query}{fragment}"


## Loader

## Load

LOADER_SCHEME = {}


class AdhocObject(object):

    def unwrap(self):
        return None

    def encode_path(self):
        if hasattr(self, "subpath") and self.subpath != "":
            path = f"{self.scheme}:{self.path}:{self.subpath}"
        else:
            path = f"{self.scheme}:{self.path}"
        pathargs = self.pathargs if hasattr(self, "pathargs") else None
        return encode_path(path, self.tag, pathargs)

    def __repr__(self):
        return self.encode_path()

    def _loaded(self, scheme, path, tag):
        if not hasattr(self, "scheme"):
            self.scheme = scheme
        if not hasattr(self, "path"):
            path, _, subpath = path.partition(":")
            self.path = path
            if not hasattr(self, "subpath") and subpath != "":
                self.subpath = subpath
        if not hasattr(self, "tag"):
            self.tag = tag

    def get(self, kwargs: dict, *adhoc_keys: str):
        pathargs = self.pathargs if hasattr(self, "pathargs") else None
        return record(kwargs, *adhoc_keys, record_to=self, record_dic=pathargs)


class AdhocLoader(object):

    def load(self, path: str, tag: str, kwargs):
        return AdhocObject()

    def register(self, scheme: str):
        global LOADER_SCHEME
        LOADER_SCHEME[scheme] = self

    def _load(self, path: str, kwargs: dict, tag: str):
        return self.load(path, tag, kwargs)

    def global_keys(self):
        return []


# def typecheck(obj, structual_type):
#     if isinstance(structual_type, str):
#         for key in list_keys(structual_type):
#             if not hasattr(obj, key):
#                 return False
#         return True
#     return isinstance(obj, structual_type)


def load(scheme: str, path: str = None, use_unwrap=False, /, **kwargs):
    global LOADER_SCHEME
    if path is None:
        scheme, _, path = scheme.partition(":")
    if scheme.startswith('_'):
        scheme = scheme[1:]
        use_unwrap = True
    with kwargs_from_path(path, **kwargs) as kwargs:
        path = kwargs.pop('_path')
        tag = kwargs.pop('_tag')
        if scheme in LOADER_SCHEME:
            obj = LOADER_SCHEME[scheme]._load(path, kwargs, tag)
            if isinstance(obj, AdhocObject):
                obj._loaded(scheme, path, tag)
            if use_unwrap:
                return obj.unwrap()
            return obj
        raise KeyError(scheme)


CLI_MAP = {}

# 関数をグローバル辞書に登録するデコレータ

def cli(func):
    global CLI_MAP
    name = func.__name__.replace('_cli', '').replace('_', '')
    CLI_MAP[name] = func
    return func


class CLIExecutor(AdhocLoader):

    def load(self, path, tag, kwargs):
        import kogitune.cli

        global CLI_MAP
        key = path.replace('_', '')
        if key in CLI_MAP:
            return CLI_MAP[key](**kwargs)
        raise KeyError(path)

CLIExecutor().register("cli")

FROM_KWARGS_MAP = {}

# 関数をグローバル辞書に登録するデコレータ
def from_kwargs(func):
    global FROM_KWARGS_MAP
    name = func.__name__.replace('_from_kwargs', '')
    FROM_KWARGS_MAP[name] = func
    return func

class FuncFromKwargsLoader(AdhocLoader):

    def load(self, path, tag, kwargs):
        global FROM_KWARGS_MAP
        path = path.lower()
        if path in FROM_KWARGS_MAP:
            return FROM_KWARGS_MAP[path](**kwargs)
        raise KeyError(path)

FuncFromKwargsLoader().register("from_kwargs")

