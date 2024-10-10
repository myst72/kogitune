from typing import List, Union
import os
import base64
import hashlib
import unicodedata

from .commons import *

class TokenizerLoader(adhoc.AdhocLoader):

    def load_from_map(self, path, kwargs):
        if path == "simple":
            return SimpleTokenizer(**kwargs)
        if path.startswith("mecab") or path.startswith("janome"):
            return JanomeTokenizer(**kwargs)
        if path.startswith("sudachi"):
            return SudachiTokenizer(**kwargs)
        return HFTokenizer(**kwargs)

TokenizerLoader({}).register("tokenizer")

class Tokenizer(adhoc.AdhocObject):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pathargs = {}

    def unique_name(self) -> str:
        return self.path

    def encode(self, text: str) -> List[int]:
        return []

    def decode(self, text: str) -> str:
        return len(text)

    def count(self, text: str) -> int:
        return len(self(text))

    def __call__(self, text: str) -> List[str]:
        return text.split(" ")


def tokenizer_base64(tokenizer_path, length=8):
    tokenizer_path = tokenizer_path.partition("?")[0]
    prefix = tokenizer_path.partition("/")[0]
    # SHA-256ハッシュを取得
    hash_object = hashlib.sha256(tokenizer_path.encode())
    hash_bytes = hash_object.digest()
    # ハッシュバイト列をBase64にエンコードし、短くする
    base64_encoded = (
        base64.urlsafe_b64encode(hash_bytes).decode().replace("=", "")[:length]
    )
    return f"{prefix}({base64_encoded})"

tokenizer_args_list = [
    "cache_dir",
    "force_download",
    "resume_download",
    "proxies",
    "use_auth_token",
    "revision",
    "subfolder",
    "use_fast",
    "tokenizer_type",
    "local_files_only",
    "trust_remote_code",
    "padding_side", 
]

def load_hftokenizer(tokenizer_path, /, **kwargs):
    transformers = adhoc.safe_import('transformers')

    if "TOKENIZERS_PARALLELISM" not in os.environ:
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

    if "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION" not in os.environ:
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

    with adhoc.kwargs_from_path(tokenizer_path, **kwargs) as kwargs:
        tokenizer_path = kwargs.pop('_path')
        args = adhoc.safe_kwargs(kwargs, tokenizer_args_list)
        # if "trust_remote_code" not in args:
        #     args["trust_remote_code"] = True
        # if "use_fast" not in args:
        #     args["use_fast"] = False
        adhoc.verbose_print('Loading a tokenizer//トークンナイザーのロード中', 
                            tokenizer_path, args, once=tokenizer_path)
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path, **args)
        except BaseException as e:
            adhoc.report_ArgumentError(
                message='Failed to load the tokenizer//トークンナイザーのロード失敗', 
                called = adhoc.function_called("AutoTokenizer.from_pretrained", 
                                            tokenizer_path, **args),
                throw=e)
    # マルチトークン拡張
    if 'mte_bases' in tokenizer.init_kwargs:
        from ..datasets.tokenizers_mte import load_mte_config
        load_mte_config(tokenizer)
    return tokenizer, args


@adhoc.from_kwargs
def _tokenizer_from_kwargs(**kwargs):
    tokenizer = kwargs.get('tokenizer', '')
    if not isinstance(tokenizer, str):
        return tokenizer

    adhoc_keys = 'tokenizer_path|tokenizer|model_path'
    use_default = adhoc.get(kwargs, 'use_default|=False')
    if use_default:
        if not isinstance(use_default, str):
            use_default = os.environ.get("TOKENIZER_PATH", "llm-jp/llm-jp-1.3b-v1.0")
        adhoc_keys = f'{adhoc_keys}|!{use_default}'
    else:
        adhoc_keys = f'{adhoc_keys}|!!'
    tokenizer_path = adhoc.get(kwargs, adhoc_keys)
    return load_hftokenizer(tokenizer_path, **kwargs)[0]

@adhoc.from_kwargs
def tokenizer_from_kwargs(**kwargs):
    return _tokenizer_from_kwargs(**kwargs)


class HFTokenizer(Tokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        path = adhoc.get(kwargs, "_subpath|_path|tokenizer_path|tokenizer|!!")
        self.tokenizer, self.pathargs = load_hftokenizer(path, **kwargs)
        self.path = self.tokenizer.name_or_path

    def unwrap(self):
        return self.tokenizer

    def unique_name(self) -> str:
        prefix = self.tokenizer.name_or_path.partition("/")[0]
        ws = [(id, w) for w, id in self.tokenizer.get_vocab().items()]
        ws.sort()
        allvoc = "".join(w for _, w in ws[:1024])
        hash_object = hashlib.sha256(allvoc.encode())
        hash_bytes = hash_object.digest()
        # ハッシュバイト列をBase64にエンコードし、短くする
        base64_encoded = (
            base64.urlsafe_b64encode(hash_bytes).decode().replace("=", "")[:8]
        )
        return f"{prefix}_{base64_encoded}"

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def decode(self, text: str) -> str:
        return len(text)

    def count(self, text: str) -> int:
        return max(len(self.tokenizer.encode(text)) - 1, 0)

    def __call__(self, text: str) -> List[str]:
        return self.tokenizer.tokenize(text)


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
