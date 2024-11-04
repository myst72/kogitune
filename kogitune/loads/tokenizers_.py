from typing import List, Union
import os
import base64
import hashlib

from .commons import *

TOKENIZER_MAP: Dict[str, Type] = {}

class TokenizerLoader(adhoc.AdhocLoader):

    def load_modules(self, path, kwargs):
        from .tokenizers_ja import SimpleTokenizer
        from .tokenizers_code import PythonTokenizer

    def load_default(self, path, kwargs):
        return HFTokenizer(**kwargs)

TokenizerLoader(TOKENIZER_MAP).register("tokenizer")

class Tokenizer(adhoc.AdhocObject):
    SCHEME='tokenizer'

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

    @classmethod
    def register(cls, names: str):
        global TOKENIZER_MAP
        for name in adhoc.list_keys(names):
            TOKENIZER_MAP[name] = cls

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
    adhoc.safe_import('tokenizers')
    transformers = adhoc.safe_import('transformers')

    if "TOKENIZERS_PARALLELISM" not in os.environ:
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

    if "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION" not in os.environ:
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

    with adhoc.kwargs_from_path(tokenizer_path, **kwargs) as kwargs:
        tokenizer_path = kwargs.pop('_path')
        args = adhoc.safe_kwargs(kwargs, tokenizer_args_list, unsafe='TOKENIZER')
        # if "trust_remote_code" not in args:
        #     args["trust_remote_code"] = True
        # if "use_fast" not in args:
        #     args["use_fast"] = False
        adhoc.verbose_print('Loading//ロード中', tokenizer_path, args, once=tokenizer_path)
        adhoc.verbose_print('強制的にオプションを追加するには、TOKENIZER_xxx=yyy', once='TOKENIZER', face='  ')
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path, **args)
        except BaseException as e:
            adhoc.report_ArgumentError(
                message='Failed to load the tokenizer//トークンナイザーのロード失敗', 
                called = adhoc.function_called("AutoTokenizer.from_pretrained", 
                                            tokenizer_path, **args),
                throw=e)

    # if 'add_special_tokens' in kwargs:
    #     print('@', tokenizer.add_special_tokens)
    #     adhoc.verbose_print('add_special_tokens =', kwargs['add_special_tokens'], once='add_specail_tokens')
    #     tokenizer.add_special_tokens =  kwargs['add_special_tokens']

    if adhoc.get(kwargs, 'reset_pad_token|=False'):
        adhoc.verbose_print('pad_tokenをリセットします', tokenizer_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

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


