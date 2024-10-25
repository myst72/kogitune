from .commons import *

TEXTEVAL_MAP = {}

class TextEvalLoader(adhoc.AdhocLoader):

    def load_modules(self, path, kwargs):
        from .textevals_basic import ByteFraction
        from .textevals_patterns import AlphaFraction

    def lower(self, path):
        return path.lower().replace('-', '_')

    def parse_loader_path(self, path, kwargs):
        if path.startswith('unique_') or path.startswith('unique-'):
            path = path[7:]
            kwargs['_unique'] = True
        return super().parse_loader_path(path, kwargs)

    def load_from_map(self, path, kwargs):
        texteval = super().load_from_map(path, kwargs)
        if "sampling" in kwargs or "sampling_length" in kwargs:
            texteval = TextSampling(texteval, **kwargs)
        return texteval

TextEvalLoader(TEXTEVAL_MAP).register("texteval")

class TextEval(adhoc.AdhocObject):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scheme = 'texteval'
        self.pathargs = {}

    def eval(self, text: str):
        return len(text)

    def __call__(self, text: str):
        return self.eval(text)

    def record_key(self):
        return self.tag if self.tag != "" else self.path

    @classmethod
    def register(cls, names: str):
        global TEXTEVAL_MAP
        for name in adhoc.list_keys(names):
            TEXTEVAL_MAP[name] = cls

TextEval.register("text_length")

"""
前処理をしています。
サンプル
"""

class TextSampling(TextEval):
    def __init__(self, texteval, **kwargs):
        super().__init__(**kwargs)
        self.get(kwargs, 'sampling|=1', 'sampling_length|=80')
        self.texteval = texteval

    def encode_as_json(self):
        return self.pathargs | self.texteval.encode_as_json()

    def eval(self, text: str):
        return self.texteval(text[:self.sampling_length])




# TODO: 未使用
# class FractionEval(adhoc.AdhocObject):
#     def __init__(self, a, b):
#         self.a = a
#         self.b = b

#     def eval(self, text: str):
#         a = self.a.eval(text)
#         b = self.b.eval(text)
#         return a if b == 0 else a / b

#     def __repr__(self):
#         return f"{self.a} / {self.b}"

#     def encode_path(self):
#         return self.a.encode_path()

