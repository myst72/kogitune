from typing import Optional, List, Union, Any
import os
import sys
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
    def __init__(self, kwargs: dict, parent = None, caller = None, open_file = None):
        self.local_kwargs = {} if kwargs is None else kwargs
        self.parent = parent
        self.caller = caller
        self.errors = kwargs.get('_errors')
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

    def items(self):
        _items = []
        for key in self.keys():
            _items.append((key, self[key]))
        return _items

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
            self.report_lazy()
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
            adhoc_print(f"List of unused arguments//未使用の引数があるよ！スペルミスない？", self.caller)
            for key in unused_keys:
                value = self[key]
                adhoc_print(f"  {key} = {repr(value)}", face='')
            if self.errors == "strict":
                raise TypeError(f"{unused_keys} is an unused keyword at {self.caller}")

    def lazy(self, key, message):
        if not hasattr(self, 'lazy_message'):
            self.lazy_message = {}
        if key not in self.lazy_message:
            self.lazy_message[key] = []
        self.lazy_message[key].append(message)

    def report_lazy(self):
        if not hasattr(self, 'lazy_message'):
            return
        for key, messages in self.lazy_message.items():
            adhoc_print(f"[{key}]")
            width = max(len(msg[0]) for msg in messages if isinstance(msg, tuple)) + 8
            for msg in messages:
                if isinstance(msg, tuple):
                    adhoc_print(colored(msg[0].ljust(width), "blue"), *msg[1:], face='  ')
                else:
                    adhoc_print(msg, face='  ')
        self.lazy_message = {}

def lazy_print(*args):
    args = [str(a) for a in args]
    if len(args) == 0:
        ARGS_STACK[0].report_lazy()
    elif len(args) == 1:
        ARGS_STACK[0].lazy('Remarks//後注', args)
    elif len(args) == 2:
        ARGS_STACK[0].lazy(args[0], args[1])
    else:
        ARGS_STACK[0].lazy(args[0], tuple(args[1:]))

## stacked

PARSE_MAP = {}

# 関数をグローバル辞書に登録するデコレータ
def parse_value(func):
    global FROM_KWARGS_MAP
    name = func.__name__.replace('parse_', '')
    PARSE_MAP[name] = func
    return func


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
    if rename_from and os.path.exists(rename_from):
        if os.path.exists(filepath):
            os.remove(filepath)
        os.rename(rename_from, filepath)
    adhoc_print('Saved Files//保存済み', filepath, desc, once=filepath, lazy=True)


def kwargs_from_stacked(caller_frame=None, /, **kwargs) -> ChainMap:
    stacked = get_stacked()
    if caller_frame is None:
        caller_frame = inspect.stack()[1].function
    kwargs.pop('caller_frame', None) # 余分なキーを消す
    return ChainMap(kwargs, parent=stacked, caller=caller_frame)


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

def dumps(data: dict, indent=0, key=None, suffix='', lines = None):
    return_string = False
    if lines is None:
        lines = []
        return_string = True
    head = ' ' * indent
    if key is not None:
        head = f'{head}"{key}": '
    if isinstance(data, (int, float, str, bool)) or data is None:
        d = json.dumps(data, ensure_ascii=False)
        lines.append(f'{head}{d}{suffix}')
    if isinstance(data, dict):
        if len(data) < 3:
            d = json.dumps(data, ensure_ascii=False)
            lines.append(f'{head}{d}{suffix}')
        else:
            lines.append(head+"{")
            for key, value in data.items():
                dumps(value, indent+2, key, ',', lines)
            lines.append("}"+suffix)
    if isinstance(data, (list, tuple)):
        if indent == 0:
            lines.append(head+"[")
            for value in data:
                dumps(value, indent+2, None, ',', lines)
            lines.append(f"]{suffix}")
        else:
            d = json.dumps(data, ensure_ascii=False)
            lines.append(f'{head}{d}{suffix}')
    if return_string:
        return '\n'.join(lines)


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

def is_once(key):
    global ONCE
    return key not in ONCE

def once(message: str, once=None):
    if once:
        once_key = once if isinstance(once, str) else message
        if once_key in ONCE:
            return ''
        ONCE[once_key] = True
    return message

def adhoc_print(*args, **kwargs):
    global ONCE, ARGS_STACK
    face = kwargs.pop("face", get_stacked("face", "🦊"))
    once = kwargs.pop("once", False)
    color = kwargs.pop("color", None)
    sep = kwargs.pop("sep", " ")
    end = kwargs.pop("end", os.linesep)
    lazy = kwargs.pop("lazy", False)
    dump_value = kwargs.pop("dump", None)
    if dump_value:
        if isinstance(dump_value, list):
            dump_value = dump_value[:3] # 3つまでにする
        args = args + (dumps(dump_value),)

    text_en = sep.join(_split_en(a) for a in args)
    text_ja = sep.join(_split_ja(a) for a in args)
    if once:
        once_key = once if isinstance(once, str) else text_en
        if once_key in ONCE:
            return
        ONCE[once_key] = True
    if lazy:
        lazy_print(*args)
        return        
    if color:
        text_en = colored(text_en, color)
        text_ja = colored(text_ja, color)
    msg = messagefy(**kwargs)
    print(f"{face}{text_en}", msg, end=end)
    if text_en != text_ja and end == os.linesep:
        print(f"{face}{text_ja}", msg, end=end)
    file = ARGS_STACK[-1].file
    if file:
        print(f"{face}{text_en}", msg, end=end, file=file)

def messagefy(**kwargs):
    ss = []
    if 'if_dislike' in kwargs:
        examples = kwargs.pop('if_dislike')
        ss.append('If you dislike ..//もしお嫌なら')
        ss.append(example_key_values(examples))
    if 'if_enforce' in kwargs:
        examples = kwargs.pop('if_enforce')
        ss.append('If you enforce to add ..//もし強制的にお加えたいなら')
        ss.append(example_key_values(examples))
    
    for key, value in kwargs.items():
        key = colored(key, 'blue')
        ss.append(f'{key} = {value}')
    return ' '.join(ss)

def example_key_values(examples:dict, color='red', face='🐼'):
    ss = []
    for key, value in examples.items():
        key = colored(key, color)
        ss.append(f'{face}{key} = {value}')
    return ', '.join(ss)

def is_verbose():
    return get_stacked('verbose', 5) > 0

def verbose_print(*args, **kwargs):
    if is_verbose():
        if 'color' not in kwargs:
            kwargs['color'] = 'cyan'
        adhoc_print(*args, **kwargs)

def is_debug():
    return get_stacked('_debug', True)


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

def exit(throw: Exception):
    if is_debug():
        raise throw
    adhoc_print("If you want to know the detail///エラーの詳細を知りたい場合は", "`_debug=True`")
    sys.exit(1)


def report_ArgumentError(message, throw, called):
#    adhoc_print(f"An error has occured.///エラーが発生しました", repr(throw))
    adhoc_print(message, colored(str(throw), 'red'))
    adhoc_print("Your parameter could be wrong.///あなたのパラメータ指定が間違っている可能性が高いです。")
    adhoc_print(called, face=' ', color='blue')
    exit(throw=throw)


## Adhoc Keys

def list_keys(keys: Union[List[str], str], sep="|"):
    if isinstance(keys, (list, tuple)):
        return keys
    suffix = ''
    if '{' in keys:  # a|b|={c|d} かなりアドホックな実装
        keys, _, suffix = keys.partition('{')
        suffix = '{' + suffix
    keys = keys.split(sep)
    keys[-1] = keys[-1]+suffix
    return keys

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
        # debug_print(keys)
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

def get_default_key(keys: List[str]):
    for key in keys:
        if not key.startswith('_'):
            return key
    return keys[0]

def panda_print(key, value):
    key = colored(key, "blue")
    adhoc_print(f"{key}={repr(value)}", face=' 🐼', once=True)

def get_adhoc(dic: dict, 
              adhoc_key: str, 
              default_value=None, 
              return_keys = True, 
              verbose=True,
              recursive=0, 
              type_fn=identity):
    keys = list_keys(adhoc_key)
    default_key = get_default_key(keys)
    for key in keys:
        matched_key = None
        if key.startswith("!!"):
            if '_path' in dic:
                path = dic['_path']
                adhoc_print(f"{path}には、`{default_key}=...`が必要です")
            exit(throw=KeyError(f"`{default_key}=...`が必要です"))
        if key.startswith("="):
            value = parse_key_value(default_key, key[1:])
            if isinstance(value, str) and value.startswith('{') and value.endswith('}'):
                value = get_adhoc(dic, value[1:-1], return_keys=False)
            if verbose and dic.get('verbose', 1):
                panda_print(default_key, value)
        elif key.startswith("!"):
            value = parse_key_value(default_key, key[1:])
            adhoc_print(
                f"Your `{default_key}` is needed.//`{default_key}`は自分で設定してよ",
                f"{default_key}={repr(value)}."
            )
        elif key in dic:
            matched_key = key
            use_stacked_key(key)
            value = dic.get(key)
        elif key[0].isupper() and key in os.environ:
            matched_key = key
            value = parse_key_value(key, os.environ[key])
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
        value, matched_key, default_key = get_adhoc(kwargs, adhoc_key)
        values.append(value)
        if isinstance(record_dic, dict) and matched_key is not None:
            record_dic[default_key] = value
        if record_to is not None:
            setattr(record_to, default_key, value)
    return values[0] if len(values) == 1 else values

def get(kwargs: dict, adhoc_key: str, default_value=None, 
        use_global=False, return_keys=False, type_fn=identity):
    if use_global:
        kwargs = get_stacked().get_kwargs() | kwargs
    return get_adhoc(kwargs, adhoc_key, default_value, return_keys=return_keys)


def get_list(kwargs: dict, adhoc_keys: str, type_fn=list_values):
    value = get(kwargs, adhoc_keys, use_global=False, return_keys=False)
    return list_values(value)

def get_words(kwargs: dict, adhoc_keys: str, type_fn=list_values):
    value, _, default_key = get(kwargs, adhoc_keys, use_global=False, return_keys=True)
    values = type_fn(value)
    if len(values) == 1 and isinstance(values[0], str):
        path = values[0]
        path, args, _ = parse_path(path, parent_args=kwargs)
        args['_readonly'] = True
        record = load('record', path, **args)
        key = get(args, 'target_key|word_key')
        if key == None:
            sample = record.samples()[0]
            key = list(sample.keys())[0]  # 先頭列
        values = []
        for sample in record.samples():
            text = sample[key]
            values.append(text)
        panda_print(default_key, values)
        return values
    return values

def safe_kwargs(kwargs: dict, adhoc_keys:List[str], unsafe=None):
    extracted = {}
    for adhoc_key in adhoc_keys:
        value, matched, default_key = get_adhoc(kwargs, adhoc_key, return_keys=True)
        if matched:
            extracted[default_key] = value
    if unsafe:
        for key, value in kwargs.items():
            if key.startswith(unsafe):
                use_stacked_key(key)
                key = key[len(unsafe)+1:]
                # adhoc_print()
                extracted[key] = value
    for key in extracted.keys():
        use_stacked_key(key)
    return extracted

# formatting

format_pattern = re.compile(r"\{([^}]+)\}")

def adhoc_format(format:str, kwargs:dict) -> list[str]:
    escaped_brace = False
    if '{{' in format and '}}' in format:
        escaped_brace = True
        format = format.replace('{{', '｛').replace('}}', '｝')
    # 正規表現を使って{}で囲まれた部分を全て抽出
    matches = format_pattern.findall(format)
    replace = {}
    for match in matches:
        if match in replace:
            continue
        if ':' in match:
            adhoc_key = match.split(':')[0]
            value = get(kwargs, adhoc_key)
            fmt = '{:' + match.split(':')[-1] + '}'
            value = fmt.format(value)
        else:
            adhoc_key = match.split(':')[0]
            value = get(kwargs, adhoc_key)
        replace[match] = value
    formatted_string = format
    for match, value in replace.items():
        formatted_string = formatted_string.replace('{'+match+'}', f'{value}')
    if escaped_brace:
        formatted_string = formatted_string.replace('｛', '{').replace('｝', '}')
    return formatted_string


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


def parse_value_of_args(kwargs: dict):
    for key in list(kwargs.keys()):
        if '__' in key:
            value = kwargs.pop(key)
            newkey, _, as_type = key.rpartition('__')
            parse_fn = load('parse_value', as_type)
            kwargs[newkey] = parse_fn(value)
    return kwargs

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

### columns




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


def kwargs_from_main(argv: List[str], use_subcommand=False, expand_config="config"):
    if use_subcommand and len(argv) > 1:
        args = parse_argv(argv[2:], expand_config=expand_config)
        args["subcommand"] = argv[1]
    else:
        args = parse_argv(argv[1:], expand_config=expand_config)
    return ChainMap(args, get_stacked(), caller='main')

## kwargs

def parse_path(path: str, parent_args:dict={}):
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
    return path, {**parent_args} | args, parsed_url.fragment

def kwargs_from_path(path: str,  /, **kwargs):
    _path, kwargs, _tag = parse_path(path, kwargs)
    kwargs = kwargs_from_stacked(path, **kwargs)
    kwargs['_path'] = _path
    kwargs['_tag'] = _tag
    return kwargs

def encode_path(path: str, tag: str, kwargs: dict):
    query = "" if kwargs is None or len(kwargs) == 0 else f"?{urlencode(kwargs)}"
    fragment = "" if tag == "" else f"#{tag}"
    return f"{path}{query}{fragment}"


## Load
LOADER_MAP = {}

def reg(name: str):
    """
    クラスを登録するデコレータ。
    """
    def decorator(cls):
        global LOADER_MAP
        if hasattr(cls, 'register'):
            cls.register(name)
        if hasattr(cls, 'SCHEME'):
            if cls.SCHEME not in LOADER_MAP:
                LOADER_MAP[cls.SCHEME] = {}
            map = LOADER_MAP[cls.SCHEME]
            for key in list_keys(name):
                map[key] = cls
        return cls
    return decorator

def find_loader_map(scheme: str):
    return LOADER_MAP.get(scheme, {})


LOADER_SCHEME = {}

class AdhocObject(object):

    def __init__(self, **kwargs):
        self.scheme = '(undefined)'
        self.path = kwargs.get('_path', '')
        self.tag = kwargs.pop('_tag', '')

    def unwrap(self):
        return None

    @property
    def upath(self):
        path = self.name if hasattr(self, "name") else self.path
        pathargs = self.pathargs if hasattr(self, "pathargs") else None
        return encode_path(path, self.tag, pathargs)

    def __repr__(self):
        return self.name if hasattr(self, "name") else self.path

    def get(self, kwargs: dict, *adhoc_keys: str):
        pathargs = self.pathargs if hasattr(self, "pathargs") else None
        return record(kwargs, *adhoc_keys, record_to=self, record_dic=pathargs)

    def load(self, scheme, path, **kwargs):
        if isinstance(path, str):
            if '|' in path or path in kwargs:
                path = self.get(kwargs, path)
        try:
            obj = load(scheme, path, **kwargs)
        except KeyError as e:
             obj = None
        setattr(self, scheme, obj)
        if hasattr(self, "pathargs") and hasattr(obj, "encode_as_json"):
            self.pathargs[scheme] = obj.encode_as_json()
        return obj

    def encode_as_json(self):
        pathargs = self.pathargs if hasattr(self, "pathargs") else {}
        return {'scheme': self.scheme, 'path': self.path} | pathargs

    def test_reload(self):
        config = self.encode_as_json()
        reloaded = load(config.copy()).encode_as_json()
        if f'{config}' == f'{reloaded}':
            return True
        print('@test_reload', config, reloaded)
        return False

    def save_config(self, filepath: str):
        adhoc_print(json.dumps(self.encode_as_json(), indent=2), face="")
        with open(filepath, "w") as w:
            json.dump(self.encode_as_json(), w, indent=2)


def load(scheme: Union[str, dict], path: Optional[Union[str, dict]] = None, /, **kwargs):
    global LOADER_SCHEME
    if isinstance(scheme, dict):
        config = scheme
        scheme = config.pop('scheme')
        path = config.pop('path')
        kwargs = config | kwargs
    if path is None:
        scheme, _, path = scheme.partition(":")
    elif isinstance(path, dict):
        config = path
        config.pop('scheme')
        path = config.pop('path')
        kwargs = config | kwargs
    use_unwrap = False
    if scheme.startswith('_'):
        scheme = scheme[1:]
        use_unwrap = True
    kwargs.pop('use_unwrap', None)
    if scheme in LOADER_SCHEME:
        obj = LOADER_SCHEME[scheme].load_from_path(path, kwargs)
        if use_unwrap:
            return obj.unwrap()
        return obj
    if '_default' in kwargs:
        return kwargs['_default']
    raise KeyError(scheme)

def load_class(class_path, check=None):
    import importlib
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    if check is not None:
        if not issubclass(cls, check):
            raise TypeError(f'{class_path} is not a subclass of {check.__name__}')
    return cls

class AdhocLoader(object):

    def __init__(self, MAP: dict):
        self.MAP = MAP
        self.not_loading = True

    def load_from_path(self, path: str, /, kwargs):
        with kwargs_from_path(path, **kwargs) as kwargs:
            path = kwargs['_path']
            path, kwargs = self.add_kwargs(path, kwargs)
            if path.startswith('class:') and '.' in path:
                cls = load_class(path[6:])
                return cls(**kwargs)
            return self.load_from_map(path, kwargs)

    def add_kwargs(self, path, kwargs):
        return path, kwargs

    def load_modules(self, path, kwargs):
        pass

    def parse_loader_path(self, path, kwargs):
        loader_path = path
        if ':' in path:
            loader_path, _, subpath = path.partition(':')
            kwargs['_subpath'] = subpath
        return self.lower(loader_path)
    
    def lower(self, loader_path):
        return loader_path.lower().replace('-', '_')

    def load_from_map(self, path, kwargs):
        loader_path = self.parse_loader_path(path, kwargs)
        if loader_path not in self.MAP:
            if self.not_loading:
                self.load_modules(path, kwargs)
                self.not_loading = False
        if loader_path in self.MAP:
            return self.MAP[loader_path](**kwargs)
        simkey = find_simkey(self.MAP, loader_path, max_distance=2)
        if simkey:
            verbose_print(f'スペルミス？ {loader_path} {simkey}')
            kwargs['_path'] = kwargs['_path'].replace(loader_path, simkey)
            return self.MAP[simkey](**kwargs)
        return self.load_default(path, kwargs)

    def parse_k_from_path(self, path, kwargs, k_string='k'):
        # 正規表現でmin後の数値をキャプチャ
        match = re.findall(r'\D*(\d+)\D*', path)
        if match:
            path = path.replace(match[0], k_string)
            kwargs[k_string] = int(match[0])
            return path
        return path

    def load_default(self, path, kwargs):
        if '_default' in kwargs:
            return kwargs['_default']
        debug_print(list(self.MAP.keys()))
        raise KeyError(path)

    def register(self, scheme: str):
        global LOADER_SCHEME
        for key in scheme.split('|'):
            LOADER_SCHEME[key] = self


CLI_MAP = {}

# 関数をグローバル辞書に登録するデコレータ

def cli(func):
    global CLI_MAP
    name = func.__name__.replace('_cli', '')
    CLI_MAP[name] = func
    name = name.lower().replace('_', '')
    if name not in CLI_MAP:
        CLI_MAP[name] = func
    return func


class CLIExecutor(AdhocLoader):

    def load_modules(self, path, kwargs):
        import kogitune.cli
        # return super().load_modules(kwargs)


CLIExecutor(CLI_MAP).register("cli")

FROM_KWARGS_MAP = {}

# 関数をグローバル辞書に登録するデコレータ
def from_kwargs(func):
    global FROM_KWARGS_MAP
    name = func.__name__.replace('_from_kwargs', '')
    FROM_KWARGS_MAP[name] = func
    name = name.lower().replace('_', '')
    if name not in CLI_MAP:
        FROM_KWARGS_MAP[name] = func
    return func

class FuncionLoader(AdhocLoader):
    pass

FuncionLoader(FROM_KWARGS_MAP).register("from_kwargs")
FuncionLoader(PARSE_MAP).register("parse_value")
