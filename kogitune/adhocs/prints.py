import os
import sys
import time
import json

try:
    from termcolor import colored
except ModuleNotFoundError:
    def colored(text, color):
        return text

### 言語

USE_JA = None

def use_ja():
    global USE_JA
    if USE_JA is None:
        USE_JA = "ja" in os.environ.get("LANG", "")
    return USE_JA

##

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
    return json.dumps(dict_as_json(data), indent=indent)


### ログ

LOGFILE_STACK = []


class LogFile(object):
    def __init__(self, filepath: str, mode="w"):
        self.filepath = filepath
        self.file = open(filepath, mode=mode)

    def __enter__(self):
        global LOGFILE_STACK
        self.file.__enter__()
        LOGFILE_STACK.append(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        global LOGFILE_STACK
        if exc_type is not None:
            pass
        LOGFILE_STACK.pop()
        return self.file.__exit__(exc_type, exc_value, traceback)

    def print(self, *args, **kwargs):
        print(*args, file=self.file, **kwargs)


def open_log_file(path: str, filename: str, mode="w"):
    os.makedirs(path, exist_ok=True)
    return LogFile(os.path.join(path, filename), mode=mode)


## フォーマット


def format_unit(num: int, scale=1000) -> str:
    """
    大きな数をSI単位系に変換して返す
    """
    if scale == 1024:
        if num < scale:
            return str(num)
        elif num < scale**2:
            return f"{num / scale:.2f}K"
        elif num < scale**3:
            return f"{num / scale**2:.2f}M"
        elif num < scale**4:
            return f"{num / scale**3:.2f}G"
        elif num < scale**5:
            return f"{num / scale**4:.2f}T"
        elif num < scale**6:
            return f"{num / scale**5:.2f}P"
        else:
            return f"{num / scale**6:.2f}Exa"
    elif scale == 60:
        if num < 1.0:
            return f"{num * 1000:.3f}ms"
        day = num // (3600 * 24)
        num = num % (3600 * 24)
        hour = num // 3600
        num = num % 3600
        min = num // 60
        sec = num % 60
        if day > 0:
            return f"{day}d{hour}h{min}m{sec:.0f}s"
        elif hour > 0:
            return f"{hour}h{min}m{sec:.0f}s"
        elif min > 0:
            return f"{min}m{sec:.0f}s"
        return f"{sec:.3f}s"
    else:
        if num < 1_000:
            return str(num)
        elif num < 1_000_000:
            return f"{num / 1_000:.2f}K"
        elif num < 1_000_000_000:
            return f"{num / 1_000_000:.2f}M"
        elif num < 1_000_000_000_000:
            return f"{num / 1_000_000_000:.2f}B"
        else:
            return f"{num / 1_000_000_000_000:.2f}T"


### プリント

WATCH_COUNT = {}
ONCE = {}

def _split_en(text, sep='//'):
    if isinstance(text, str):
        return text.split(sep)[0]
    return str(text)

def _split_ja(text, sep='//'):
    if isinstance(text, str):
        return text.split(sep)[-1]
    return str(text)

def list_kwargs(**kwargs):
    ss = []
    for key, value in kwargs.items():
        ss.append(f"{key}={value}")
    return ss

def aargs_print(*args, **kwargs):
    global ONCE
    face = kwargs.pop("face", "🦊")
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
    if len(LOGFILE_STACK) > 0:
        LOGFILE_STACK[-1].print(f"{face}{text_en}", end=end)


def notice(*args, **kwargs):
    aargs_print(*args, *list_kwargs(**kwargs))
    sys.stdout.flush()

def warn(*args, **kwargs):
    aargs_print(colored("FIXME" "red"), *args, *list_kwargs(**kwargs))


def notice_kwargs(path, kwargs, exception=None):
    if isinstance(exception, BaseException):
        aargs_print(repr(exception))
    aargs_print(path, kwargs)


SAVED_LIST = []


def saved(filepath: str, desc: str, rename_from=None):
    global SAVED_LIST
    if rename_from and os.path.exists(rename_from):
        if os.path.exists(filepath):
            os.remove(filepath)
        os.rename(rename_from, filepath)
    if filepath not in [file for file, _ in SAVED_LIST]:
        SAVED_LIST.append((filepath, desc))


def report_saved_files():
    global SAVED_LIST
    if len(SAVED_LIST) == 0:
        return
    width = max(len(filepath) for filepath, _ in SAVED_LIST) + 8
    for filepath, desc in SAVED_LIST:
        print(colored(filepath.ljust(width), "blue"), desc)
    SAVED_LIST = []


class start_timer(object):
    """
    タイマー
    """

    def __init__(self):
        pass

    def __enter__(self):
        self.start_time = time.time()
        return self

    def notice(self, *args, iteration="total", **kwargs):
        elapsed_time = time.time() - self.start_time
        total = kwargs.get(iteration, None)
        if total is not None and total > 0:
            kwargs = dict(
                elapsed_time=format_unit(elapsed_time, scale=60),
                elapsed_second=round(elapsed_time, 3),
                throughput=round(elapsed_time / total, 3),
                iteration=total,
                **kwargs,
            )
        else:
            kwargs = dict(
                elapsed_time=format_unit(elapsed_time, scale=60),
                elapsed_second=round(elapsed_time, 3),
                **kwargs,
            )
        notice(*args, **kwargs)

    def __exit__(self, exc_type, exc_value, traceback):
        pass


def describe_counters(
    counters, caption, output_file, columns=["Item", "Count", "%"], head=20
):
    import pandas as pd

    df = pd.DataFrame.from_dict(counters, orient="index").reset_index()
    df.columns = columns[:2]
    total_count = df[columns[1]].sum()
    df[columns[2]] = (df[columns[1]] / total_count) * 100
    df[columns[2]] = df[columns[2]].round(3)
    aargs_print(df.head(head), face="")
    aargs_print(f"{caption} See {output_file}")
