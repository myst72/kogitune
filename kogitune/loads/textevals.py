from .commons import *

TEXTEVAL_MAP = {}

class TextEvalLoader(adhoc.AdhocLoader):

    def load_modules(self, path, kwargs):
        from .textevals_basic import AlphaFraction

    def lower(self, path):
        return path.lower().replace('_', '-')

    def load_from_map(self, path, kwargs):
        texteval = super().load_from_map(path, kwargs)
        # if "fraction" in kwargs:
        #     path, tag, kwargs = adhoc.parse_path(kwargs.pop("fraction"))
        #     fraction = self.load(path, tag, **kwargs)
        #     return FractionEval(texteval, fraction)
        return texteval

TextEvalLoader(TEXTEVAL_MAP).register("texteval")

class TextEval(adhoc.AdhocObject):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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

TextEval.register("text-length|charcter-length|charcter")

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

