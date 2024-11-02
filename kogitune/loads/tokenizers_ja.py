import unicodedata

from .commons import *
from .tokenizers_ import Tokenizer

@adhoc.reg('janome')
class JanomeTokenizer(Tokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "janome"
        #self.pathargs = {}
        janome = adhoc.safe_import('janome.tokenizer', 'janome')
        self.janome = janome.Tokenizer()

    def __call__(self, text: str) -> List[str]:
        tokens = [token.surface for token in self.janome.tokenize(text)]
        return tokens


@adhoc.reg('sudachipy')
class SudachiTokenizer(Tokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "sudachipy"
        # self.pathargs = {}
        adhoc.safe_import('sudachipy', "sudachipy sudachidict_core")
        dictionary = adhoc.safe_import('sudachipy.dictionary', 'sudachidict_core')
        tokenizer = adhoc.safe_import('sudachipy.tokenizer', 'sudachidict_core')
        self.sudachi = dictionary.Dictionary().create()
        self.mode = tokenizer.Tokenizer.SplitMode.C

    def __call__(self, text: str) -> List[str]:
        tokens = [m.surface() for m in self.sudachi.tokenize(text, self.mode)]
        return tokens

##
# 文字の種類によるトークンナイザー

def char_type(char):
    if ord(char) < 256:
        if char.isalpha():
            return "ALPHA"
        if char.isdigit():
            return "DIGIT"
        if char in "+-*/=<>|&~^_":
            return "OP"
        return char
    else:
        cat = unicodedata.category(char)
        name = unicodedata.name(char)
        if cat.startswith("P"):
            return "PUNCT"
        if cat.startswith("S"):
            return "EMOJI"
        if name.startswith("HIRAGANA"):
            return "HIRAGANA"
        if name.startswith("KATAKANA"):
            return "KATAKANA"
        if name.startswith("CJK UNIFIED IDEOGRAPH"):
            return "KANJI"
        if name.startswith("FULL"):
            return "FULL"
        return "OTHER"

def simple_tokenize(text):
    token = []
    result = []

    def append():
        nonlocal token, result
        if len(token) > 0:
            s = "".join(token)
            if s != " ":
                result.append(s)
            token = []

    prev_type = None
    for char in text:
        current_type = char_type(char)
        if prev_type and current_type != prev_type:
            if len(token) > 0:
                append()
        token.append(char)
        prev_type = current_type
    append()
    return result


@adhoc.reg('simple')
class SimpleTokenizer(Tokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "simple"

    def __call__(self, text: str) -> List[str]:
        token = []
        result = []

        def append():
            nonlocal token, result
            if len(token) > 0:
                s = "".join(token)
                if s != " ":
                    result.append(s)
                token = []

        prev_type = None
        for char in text:
            current_type = char_type(char)
            if prev_type and current_type != prev_type:
                if len(token) > 0:
                    append()
            token.append(char)
            prev_type = current_type
        append()
        return result


@adhoc.reg('simple')
class PythonTokenizer(Tokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "python"

    def __call__(self, text: str) -> List[str]:
        token = []
        result = []

        def append():
            nonlocal token, result
            if len(token) > 0:
                s = "".join(token)
                if s != " ":
                    result.append(s)
                token = []

        prev_type = None
        for char in text:
            current_type = char_type(char)
            if prev_type and current_type != prev_type:
                if len(token) > 0:
                    append()
            token.append(char)
            prev_type = current_type
        append()
        return result



