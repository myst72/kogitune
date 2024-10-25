import json

from .commons import *

FILTER_MAP = {}

class TextFilterLoader(adhoc.AdhocLoader):

    def load_modules(self, path, kwargs):
        from .filters_docs import UnicodeNormalization

    def load_from_map(self, path, kwargs):
        global FILTER_MAP
        if path.endswith(".json"):
            return adhoc.load_adhoc_config(path, **kwargs)
        return super().load_from_map(path, kwargs)
    
    def load_default(self, path, kwargs):
        try:
            filter = MaxMinFilter(**kwargs)
            return filter
        except KeyError:
            pass
        return super().load_default(path, kwargs)

TextFilterLoader(FILTER_MAP).register("filter")


class TextFilter(adhoc.AdhocObject):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scheme = 'filter'
        self.target = self.get(kwargs, "_target|target|text_key|=text")
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

    @classmethod
    def register(cls, names: str):
        global FILTER_MAP
        for name in adhoc.list_keys(names):
            FILTER_MAP[name] = cls

TextFilter.register('none')

@adhoc.from_kwargs
def filter_from_kwargs(**kwargs):
    filter = kwargs.pop("filter", None)
    if filter is None:
        filter = TextFilter(**kwargs)
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
        self.path = 'maxmin'
        self.get(
            kwargs,
            "max_inclusive|max",
            "min_inclusive|min",
            "max_exclusive", "min_exclusive",
        )
        self.load("texteval", '_subpath|texteval|_path', **kwargs)
        self.name = f'maxmin:{self.texteval.path}'
        self.record_key = self.texteval.path

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

MaxMinFilter.register("maxmin")

class ContainsFilter(TextFilter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.load("pattern", '_subpath|pattern|_path|!!', **kwargs)
        self.name = f'contains:{self.pattern.path}'

    def filter(self, sample: dict) -> Optional[str]:
        text = sample[self.target]
        if self.pattern.contains(text):
            return sample
        return None

ContainsFilter.register('contains')

class ReplaceFilter(TextFilter):
    """
    置き換えフィルター
    a = 'replace:url#<URL>|date#<date>'
    """

    def __init__(self, **kwargs):
        """
        置き換えフィルターを作る
        :param patterns: 置き換える文字列パターンのリスト
        """
        super().__init__(**kwargs)
        self.patterns = self.get(kwargs, "_subpath|patterns|!!")
        self.replace_patterns = []
        for pat in adhoc.list_keys(self.patterns):
            pat, sep, replaced = pat.partition('#')
            if sep == '':
                replaced = pat.split('_')[0]
                replaced = f'<{replaced}>'
            pattern = adhoc.load('pattern', pat, _replaced=replaced)
            self.replace_patterns.append(pattern)

    def filter_text(self, text)->Optional[str]:
        for pattern in self.replace_patterns:
            text = pattern.replace(text)
        return text

ReplaceFilter.register('replace')

## Composition

class ComposeFilter(TextFilter):
    """
    テキストフィルタを合成する
    :param filters:
    """

    def __init__(self, **kwargs):
        """
        評価関数フィルタを作る
        """
        super().__init__(**kwargs)
        self.path = "compose"
        self.filters = [
            load_filter(f) 
            for f in adhoc.get(kwargs, "filters", [])
        ]

    def encode_as_json(self):
        return {
            "scheme": "filter",
            "path": self.path,
            "filters": [f.encode_as_json() for f in self.filters],
        } | self.pathargs

    def filter(self, sample: dict) -> Optional[dict]:
        for filter in self.filters:
            text = filter(dict)
            if text is None:
                return None
        return sample

ComposeFilter.register("compose")

def load_filter(config, target='text'):
    if isinstance(config, TextFilter):
        return config
    if isinstance(config, str):
        return adhoc.load('filter', config)
    assert isinstance(config, dict)
    return adhoc.load('filter', config['path'], **config)

class ChoiceFilter(ComposeFilter):
    """
    テキストフィルタを合成する
    :param filters:
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def filter(self, sample:dict) -> Optional[dict]:
        for f in self.filters:
            sample2 = f(sample)
            if sample2 is not None:
                return sample2
        return None

ChoiceFilter.register('choice|choose')

class ExtractFilter(ComposeFilter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.extractor = self.load(kwargs, '_subpath|extractor|!!')

    def filter(self, sample:dict) -> Optional[dict]:
        text = sample[self.target]
        text = '\n'.join(self.extractor.extract(text))
        sample[self.target] = text
        return super().filter(sample)

ChoiceFilter.register('extract')

