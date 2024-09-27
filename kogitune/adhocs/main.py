from typing import List
import os
import sys
import re
import inspect
import importlib

from .dicts import (
    parse_key_value,
    load_config,
    list_keys, list_values,
    ChainMap,
    use_os_environ,
    find_dict_from_keys,
    extract_dict_with_keys,
    extract_dict_with_prefix,
    parse_path,
    encode_path,
)

from .prints import aargs_print, saved, report_saved_files

# main


class AdhocArguments(ChainMap):
    """
    アドホックな引数パラメータ
    """

    def __init__(self, args: dict, parent=None, caller="main"):
        super().__init__(args, parent)
        self.caller = caller
        self.errors = "ignore"
        self.saved_files = False

    def __enter__(self):
        push_stack_aargs(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.check_unused()
        if self.saved_files:
            report_saved_files()
            self.saved_files = False
        pop_stack_aargs()

    def saved(self, filepath: str, desc: str, rename_from=None):
        self.saved_files = True
        saved(filepath, desc, rename_from=rename_from)

    def check_unused(self):
        unused_keys = self.unused_keys()
        if len(unused_keys) > 0:
            if self.errors == "ignore":
                return
            if self.errors == "strict":
                raise TypeError(f"{key} is an unused keyword at {self.caller}")
            aargs_print(f"未使用の引数があるよ//List of unused arguments")
            for key in unused_keys:
                value = self[key]
                print(f"{key}: {repr(value)}")
            print(f"[確認↑] スペルミスはない？//Check if typos exist.")

    def from_kwargs(self, **kwargs):
        return AdhocArguments(kwargs, self)


### parse_argument

_key_pattern = re.compile(r"^[A-Za-z0-9\.\-_]+\=")


def _parse_key_value(key, next_value, args):
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
        key, value = _parse_key_value(arg, next_value, args)
        if value is not None:
            key = key.replace("-", "_")
            if key == expand_config:
                loaded_data = load_config(value)
                args.update(loaded_data)
            else:
                args[key.replace("-", "_")] = value
    del args["_"]
    return args


def parse_main_args(use_subcommand=False, use_environ=True, expand_config="config"):
    env = use_os_environ() if use_environ else None
    if use_subcommand and len(sys.argv) > 1:
        args = parse_argv(sys.argv[2:], expand_config=expand_config)
        args["subcommand"] = sys.argv[1]
    else:
        args = parse_argv(sys.argv[1:], expand_config=expand_config)
    aargs = AdhocArguments(args, env, caller="main")
    aargs.errors = "main"
    return aargs


def load_symbol(module_path, symbol):
    module = importlib.import_module(module_path)
    return getattr(module, symbol)


def load_class(class_path, check=None):
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    if check is not None:
        if not issubclass(cls, check):
            raise TypeError(f"{class_path} is not a subclass of {check.__name__}")
    return cls


def instantiate_from_dict(dic: dict, check=None):
    class_path = dic.pop("class_path")
    if class_path is None:
        raise TypeError(f"No class_path in {dic}")
    cls = load_class(class_path, check=check)
    args = dic.pop("args", [])
    kwargs = dic.pop("kwargs", {})
    # print('@@', cls, f'args={args}', f'kwargs={kwargs}')
    return cls(*args, **kwargs)


def load_subcommand(subcommand, **kwargs):
    fname = f"{subcommand}_cli"
    if "." in fname:
        cls = load_class(fname)
    else:
        cls = load_symbol("kogitune.cli", fname)
    cls(**kwargs)


def launch_subcommand(module_path="kogitune.cli"):
    with parse_main_args(use_subcmd=True) as aargs:
        subcmd = aargs["subcommand"]
        fname = f"{subcmd}_cli"
        if "." in fname:
            cls = load_class(fname)
        else:
            cls = load_symbol(module_path, fname)
        cls()


###
AARGS_STACKS = []
AARGS_ENV = None


def get_stack_aargs():
    global AARGS_STACKS, AARGS_ENV
    if len(AARGS_STACKS) == 0:
        if AARGS_ENV is None:
            AARGS_ENV = AdhocArguments({}, use_os_environ())
        return AARGS_ENV
    return AARGS_STACKS[-1]


def push_stack_aargs(aargs):
    global AARGS_STACKS
    AARGS_STACKS.append(aargs)


def pop_stack_aargs():
    global AARGS_STACKS
    AARGS_STACKS.pop()


def from_kwargs(**kwargs) -> AdhocArguments:
    if "aargs" in kwargs:
        # aargs をパラメータに渡すのは廃止
        raise ValueError("FIXME: aargs is unncessary")
    aargs = get_stack_aargs()
    caller_frame = inspect.stack()[1].function
    return AdhocArguments(kwargs, parent=aargs, caller=caller_frame)


def aargs_from(args: dict = None, **kwargs) -> AdhocArguments:
    if args is None:
        args = kwargs
    else:
        args = args | kwargs
    aargs = get_stack_aargs()
    caller_frame = inspect.stack()[1].function
    return AdhocArguments(args, parent=aargs, caller=caller_frame)


def get(kwargs: dict, keys: str, use_global=False, with_key=False):
    if use_global:
        kwargs = {**get_stack_aargs()} | kwargs
    key, value = find_dict_from_keys(kwargs, list_keys(keys))
    return (key, value) if with_key else value

def get_list(kwargs: dict, keys: str, use_global=False):
    value = get(kwargs, keys, use_global=use_global, with_key=False)
    return list_values(value)


def get_global_kwargs(global_keys: List[str], **kwargs) -> dict:
    aargs = get_stack_aargs()
    if global_keys is None:
        return {**aargs} | kwargs
    return extract_dict_with_keys(aargs, global_keys) | kwargs


def is_verbose():
    aargs = get_stack_aargs()
    return aargs["verbose|=True"]


def verbose_print(*args, **kwargs):
    aargs = get_stack_aargs()
    if aargs["verbose|=True"]:
        if 'color' not in kwargs:
            kwargs['color'] = 'cyan'
        aargs_print(*args, **kwargs)


## Load

LOADER_SCHEME = {}


class LoaderObject(object):

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

    def get(self, kwargs: dict, *keys_list: str):
        assert isinstance(kwargs, dict)
        values = []
        for key in keys_list:
            _, _, default_value = key.partition("|=")
            if default_value == "":
                default_value = "None"
            recordable = True
            if key.startswith("_"):  # _で始まるときは残さない
                recordable = False
                key = key[1:]
            default_key = key.split("|")[0]
            if "_path" in key and hasattr(self, "subpath") and self.subpath == "":
                default_key = "subpath"
            key, value = find_dict_from_keys(kwargs, list_keys(key))
            if recordable:
                if (
                    hasattr(self, "pathargs")
                    and default_key != "subpath"
                    and str(value) != default_value
                ):
                    self.pathargs[default_key] = value
                setattr(self, default_key, value)
            values.append(value)
        return values[0] if len(values) == 1 else values



class AdhocLoader(object):

    def load(self, path: str, tag: str, kwargs):
        return LoaderObject()

    def register(self, scheme: str):
        global LOADER_SCHEME
        LOADER_SCHEME[scheme] = self

    def _load(self, path: str, kwargs: dict, tag: str):
        kwargs = get_global_kwargs(self.global_keys(), **kwargs)

        return self.load(path, tag, kwargs)

    def global_keys(self):
        return []


def typecheck(obj, structual_type):
    if isinstance(structual_type, str):
        for key in list_keys(structual_type):
            if not hasattr(obj, key):
                return False
        return True
    return isinstance(obj, structual_type)


def load(
    scheme: str, path: str = None, /,
    extract_prefix: str = None, use_unwrap=False, **kwargs,
):
    global LOADER_SCHEME
    if path is None:
        scheme, _, path = scheme.partition(":")
    if scheme.startswith('_'):
        scheme = scheme[1:]
        use_unwrap = True
    # if extract_prefix is not None:
    #     kwargs = extract_dict_with_prefix(kwargs, extract_prefix)
    path, args, tag = parse_path(path, parent_args=kwargs)
    if scheme in LOADER_SCHEME:
        obj = LOADER_SCHEME[scheme]._load(path, args, tag)
        if isinstance(obj, LoaderObject):
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
        import kogitune.metrics.cli

        global CLI_MAP
        path = path.replace('_', '')
        if path in CLI_MAP:
            return CLI_MAP[path](**kwargs)
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


WORDLIST_MAP = {}

def wordlist(func):
    global WORDLIST_MAP
    WORDLIST_MAP[func.__name__] = func
    return func

class WordListLoader(AdhocLoader):

    def load(self, path, tag, kwargs):
        global WORDLIST_MAP

        path = path.lower()

        if path in WORDLIST_MAP:
            return WORDLIST_MAP[path]()
        raise KeyError(path)

WordListLoader().register("wordlist")
