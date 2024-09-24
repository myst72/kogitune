from typing import List, Union
import os
import base64
import hashlib
import unicodedata

from .commons import *


class Tokenizer(adhoc.LoaderObject):
    def __init__(self, path, kwargs):
        self.path = path
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


class TokenizerLoader(adhoc.AdhocLoader):

    def load(self, path, tag, kwargs):
        if path == "simple":
            return SimpleTokenizer(path, kwargs)
        if path.startswith("mecab") or path.startswith("janome"):
            return JanomeTokenizer(path, kwargs)
        if path.startswith("sudachi"):
            return SudachiTokenizer(path, kwargs)
        return HFTokenizer(path, kwargs)


TokenizerLoader().register("tokenizer")


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

tokenizer_available_keys = [
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

def load_hftokenizer(tokenizer_path, **kwargs):
    from transformers import AutoTokenizer

    if "TOKENIZERS_PARALLELISM" not in os.environ:
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

    tokenizer_path, args, _ = adhoc.parse_path(tokenizer_path, parent_args=kwargs)
    args = adhoc.safe_kwargs(args, *tokenizer_available_keys)
    if "trust_remote_code" not in args:
        args["trust_remote_code"] = True
    if "use_fast" not in args:
        args["use_fast"] = False
    try:
        adhoc.print('Loading..//ロード中', tokenizer_path, args, once=True)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **args)
    except BaseException as e:
        adhoc.notice_kwargs(tokenizer_path, args, exception=e)
    return tokenizer, args

@adhoc.from_kwargs
def tokenizer_from_kwargs(**kwargs):
    tokenizer = kwargs.get('tokenizer', '')
    if not isinstance(tokenizer, str):
        return tokenizer
    keys = 'tokenizer_path|tokenizer|model_path'
    use_default = kwargs.get('use_default', False)
    if use_default:
        if not isinstance(use_default, str):
            use_default = os.environ.get("DEFAULT_TOKENIZER_PATH", "llm-jp/llm-jp-1.3b-v1.0")
        keys = f'{keys}|!{use_default}'
    else:
        keys = f'{keys}|!!'
    tokenizer_path = adhoc.get(kwargs, keys)
    return load_hftokenizer(tokenizer_path, **kwargs)[0]


class HFTokenizer(Tokenizer):
    def __init__(self, path, kwargs):
        kwargs.pop("tokenizer_path", None)
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
        return self.tokenize(text)


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
    def __init__(self, path, kwargs):
        self.path = path
        self.pathargs = {}

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
    def __init__(self, path, kwargs):
        self.path = path
        self.pathargs = {}
        janome = adhoc.safe_import('janome')
        self.janome = janome.tokenizer.Tokenizer()

    def __call__(self, text: str) -> List[str]:
        tokens = [token.surface for token in self.janome.tokenize(text)]
        return tokens


class SudachiTokenizer(Tokenizer):
    def __init__(self, path, kwargs):
        self.path = path
        self.pathargs = {}
        sudachipy = adhoc.safe_import('sudachipy', "sudachipy sudachidict_core")

        self.sudachi = sudachipy.dictionary.Dictionary().create()
        self.mode = sudachipy.tokenizer.Tokenizer.SplitMode.C

    def __call__(self, text: str) -> List[str]:
        tokens = [m.surface() for m in self.sudachi.tokenize(text, self.mode)]
        return tokens
