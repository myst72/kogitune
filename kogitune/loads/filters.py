import json

from .commons import *

FILTER_MAP = {}

class TextFilterLoader(adhoc.AdhocLoader):

    def load_from_map(self, path, kwargs):
        global FILTER_MAP
        if path.endswith(".json"):
            ## TODO
            return load_filter_config(path, **kwargs)
        return super().load_from_map(path, kwargs)
    
    def load_default(self, path, kwargs):
        try:
            print("@maxmin", kwargs)
            filter = MaxMinFilter(**kwargs)
            return filter
        except KeyError:
            pass
        return super().load_default(path, kwargs)

TextFilterLoader(FILTER_MAP).register("filter")

class TextFilter(adhoc.AdhocObject):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.target = self.get(kwargs, "target|text_key|=text")
        self.pathargs = {}

    def filter(self, sample: dict) -> Optional[dict]:
        sample[self.target] = self.filter_text(sample[self.target])
        return sample

    def filter_text(self, text:str) -> Optional[str]:
        return text

    def __call__(self, iterable):
        for sample in iterable:
            if self.filter(sample):
                yield sample

    def filter_list(self, samples:List[dict]):
        return [sample for sample in samples if self.filter(sample)]


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


@adhoc.from_kwargs
def filter_from_kwargs(**kwargs):
    filter = kwargs.pop("filter", None)
    if filter is None:
        filter = TextFilter(**{"_path": "none"})
    else:
        filter = adhoc.load("filter", filter, extract_prefix="filter", **kwargs)
    return filter


class MaxMinFilter(TextFilter):
    """
    評価関数の最大値と最小値からフィルターする
    """

    def __init__(self, **kwargs):
        """
        評価関数フィルタを作る
        """
        super().__init__(**kwargs)
        self.get(
            kwargs,
            "max_inclusive|max",
            "min_inclusive|min",
            "max_exclusive", "min_exclusive",
        )
        path = adhoc.get(kwargs, '_subpath|_path')
        self.texteval = adhoc.load("texteval", path, **kwargs)
        self.record_key = self.texteval.path
        self.path = f'maxmin:{path}'

    def filter(self, sample: dict) -> Optional[str]:
        text = sample[self.target]
        value = self.texteval(text)
        if self.min_inclusive and self.min_inclusive > value:
            return None
        if self.max_inclusive and self.max_inclusive < value:
            return None
        if self.min_exclusive and self.min_exclusive >= value:
            return None
        if self.min_exclusive and self.min_exclusive <= value:
            return None
        sample[self.record_key] = round(value, 4)
        return sample

MaxMinFilter.register("maxmin|max|min")

class ContainsFilter(TextFilter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        path = adhoc.get(kwargs, '_subpath|_path')
        self.pattern = adhoc.load("pattern", path, **kwargs)
        self.path = f'contains:{path}'

    def filter(self, sample: dict) -> Optional[str]:
        text = sample[self.target]
        if self.pattern.contains(text):
            return sample
        return None

ContainsFilter.register('contains')


## Composition

class CompositeFilter(TextFilter):
    """
    テキストフィルタを合成する
    :param filters:
    """

    def __init__(self, **kwargs):
        """
        評価関数フィルタを作る
        """
        super().__init__(**kwargs)
        self.filters = adhoc.get(kwargs, "filters", [])

    def encode_as_json(self):
        return {
            "type": self.path,
            "filters": [f.encode_as_json() for f in self.filters],
            "kwargs": self.pathargs,
        }

    def filter(self, sample: dict) -> Optional[dict]:
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

