from typing import Optional, List, Union, Any
import json

from .commons import *

FILTER_MAP = {}


class TextFilter(adhoc.AdhocObject):
    def __init__(self, name: str, subpath: str, kwargs):
        self.path = name
        self.subpath = subpath
        self.target = self.get(kwargs, "target|=text")
        self.pathargs = {}

    def filter(self, sample: dict) -> Optional[dict]:
        return sample

    def __call__(self, sample: dict):
        return self.filter(sample)

    def encode_as_json(self):
        return self.encode_path()

    def save_config(self, filepath: str):
        adhoc.print(json.dumps(self.encode_as_dict(), indent=2), face="")
        with open(filepath, "w") as w:
            json.dump(self.encode_as_json(), w, indent=2)

    @classmethod
    def register(cls, names: str):
        global FILTER_MAP
        for name in adhoc.list_keys(names):
            FILTER_MAP[name] = cls


class TextFilterLoader(adhoc.AdhocLoader):

    def load(self, path, tag, kwargs):
        global FILTER_MAP
        path, _, subpath = path.partition(":")
        if path.endswith(".json"):
            ## TODO
            return load_filter_config(path, **kwargs)
        if "." in path:
            func = adhoc.load_class(path)
            if not issubclass(func, TextFilter):
                raise TypeError(f"{path} is not a subclass of TextEval")
            return func(path, subpath, kwargs)
        path = path.lower().replace("_", "-")
        if path in FILTER_MAP:
            filter = FILTER_MAP[path](path, subpath, kwargs)
            return filter
        try:
            subpath = f"{path}:{subpath}"
            print("@subpath", subpath)
            filter = MaxMinFilter(path, subpath, kwargs)
            return filter
        except KeyError:
            pass
        raise KeyError(path)


TextFilterLoader().register("filter")

@adhoc.from_kwargs
def filter_from_kwargs(**kwargs):
    filter = kwargs.pop("filter", None)
    if filter is None:
        filter = TextFilter('nop', '', kwargs)
    else:
        filter = adhoc.load("filter", filter, extract_prefix="filter", **kwargs)
    return filter


class CompositeFilter(TextFilter):
    """
    テキストフィルタを合成する
    :param filters:
    """

    def __init__(self, path, subpath, kwargs):
        """
        評価関数フィルタを作る
        """
        super().__init__("compose", subpath, kwargs)
        self.filters = kwargs.get("filters", [])

    def encode_as_json(self):
        return {
            "type": self.path,
            "filters": [f.encode_as_json() for f in self.filters],
            "kwargs": self.pathargs,
        }

    def __call__(self, sample: dict) -> Optional[dict]:
        for filter in self.filters:
            text = filter(dict)
            if text is None:
                return None
        return sample


CompositeFilter.register("compose")


def generate_filter(config: dict, kwargs):
    if isinstance(config, dict):
        path = config["type"]
        kwargs = config["kwargs"] | kwargs
        kwargs.pop("filters", None)
        filters = [generate_filter(conf, kwargs) for conf in config["filters"]]
        filter = adhoc.load("filter", path, filters=filters, **kwargs)
        return filter
    assert isinstance(config, str)
    path, args, tag = adhoc.parse_path(config)
    return adhoc.load("filter", path, **(args | kwargs))


def load_filter_config(config_file, **kwargs):
    with open(config_file) as f:
        config = json.load(f)
        return generate_filter(config, kwargs)


class MaxMinFilter(TextFilter):
    """
    評価関数の最大値と最小値からフィルターする
    """

    def __init__(self, path, subpath, kwargs):
        """
        評価関数フィルタを作る
        """
        super().__init__("maxmin", subpath, kwargs)
        self.get(
            kwargs,
            "max_inclusive|max",
            "min_inclusive|min",
        )
        self.texteval = adhoc.load("texteval", subpath, **kwargs)
        self.record_key = self.texteval.path

    def filter(self, sample: dict) -> Optional[str]:
        text = sample[self.target]
        value = self.texteval(text)
        if self.min_inclusive and self.min_inclusive > value:
            # adhoc.print(
            #     f"[DROP] {value} < min={self.min_inclusive} {repr(text)}",
            #     watch=self.record_key,
            # )
            return None
        if self.max_inclusive and self.max_inclusive < value:
            # adhoc.print(
            #     f"[DROP] {value} > max={self.max_inclusive} {repr(text)}",
            #     watch=self.record_key,
            # )
            return None
        sample[self.record_key] = round(value, 4)
        return sample


MaxMinFilter.register("maxmin|max|min")
