from .commons import *
from .textevals import TextEval

import re
import zlib
from collections import Counter
import math

class ByteCount(TextEval):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.name = "byte-length"
        self.get(kwargs, "encoding|=utf-8")

    def eval(self, text: str) -> int:
        return len(text.encode(self.encoding, errors="ignore"))


ByteCount.register("byte-length|byte-count|byte")


class UniqueByteCount(TextEval):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.name = "unique-byte-length"
        self.get(kwargs, "encoding|=utf-8")

    def eval(self, text: str) -> int:
        return len(set(text.encode(self.encoding, errors="ignore")))


UniqueByteCount.register("unique-byte-length|unique-byte-count|unique-byte")


class ByteFraction(TextEval):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.name = "byte-fraction"
        self.get(kwargs, "encoding|=utf-8")

    def eval(self, text: str) -> int:
        a = len(text.encode(self.encoding, errors="ignore"))
        b = len(text)
        return a / b if b != 0 else 1


ByteFraction.register("byte-fraction")


class ZlibCount(TextEval):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.name = "zlib"
        self.get(kwargs, "encoding|=utf-8")

    def eval(self, text: str) -> int:
        encoded = text.encode(self.encoding, errors="ignore")
        compressed = zlib.compress(encoded, level=9)
        return len(compressed)


ZlibCount.register("zlib-length|zlib-count|zlib")


class ZlibFraction(TextEval):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.name = "zlib-fraction"
        self.get(kwargs, "encoding|=utf-8")

    def eval(self, text: str) -> int:
        encoded = text.encode(self.encoding, errors="ignore")
        compressed = zlib.compress(encoded, level=9)
        b = len(text)
        return len(compressed) / b if b != 0 else 1


ZlibFraction.register("zlib-fraction")


class AlphaCount(TextEval):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.name = "alpha-count"
        self.get(kwargs, "regex|=[A-z]")
        self.pattern = re.compile(self.regex)

    def eval(self, text: str) -> int:
        return len(self.pattern.findall(text))


AlphaCount.register("alpha-count|alpha")


class AlphaFraction(AlphaCount):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.name = "alpha-fraction"
        self.get(kwargs, "regex|=[A-z]")
        self.pattern = re.compile(self.regex)

    def eval(self, text: str) -> int:
        a = len(self.pattern.findall(text))
        b = len(text)
        return a / b if b != 0 else 1

AlphaFraction.register("alpha-fraction")


class TokenCount(TextEval):
    def __init__(self, **kwargs) -> None:
        super().__init__(**(kwargs | {"_path": "token"}))
        subpath = self.get(kwargs, "_subpath|tokenizer_path|tokenizer|!!")
        self.tokenizer = adhoc.load("tokenizer", subpath, **kwargs)

    def eval(self, text: str) -> int:
        return self.tokenizer.count(text)


TokenCount.register("token|token-count")


class TokenFraction(TextEval):
    def __init__(self, **kwargs) -> None:
        super().__init__(**(kwargs | {"_path": "token-fraction"}))
        subpath = self.get(kwargs, "_subpath|tokenizer_path|tokenizer|!!")
        self.tokenizer = adhoc.load("tokenizer", subpath, **kwargs)

    def eval(self, text: str) -> int:
        a = self.tokenizer.count(text)
        b = len(text)
        return a / b if b != 0 else 1


TokenFraction.register("token-fraction")


class TokenEntropy(TextEval):
    """
    任意のトークンリストのエントロピーを計算でき、それによりトークンの分布がどの程度多様か、
    またはどの程度予測可能かが分かります。
    エントロピーが高いほど、トークンの分布は多様で予測が難しいと言えます。
    逆にエントロピーが低い場合、トークンの分布は比較的均一で予測が容易です。
    :param tokenizer:
    """

    def __init__(self, **kwargs):
        """
        トークンナイザーによるエントロピー評価関数を作る
        :param tokenizer: トークンナイザー(もしくはトークンナイザー名)
        """
        super().__init__(**(kwargs | {"_path": "token-entropy"}))
        subpath = self.get(kwargs, "__subpath|tokenizer_path|tokenizer|!!")
        self.tokenizer = adhoc.load("tokenizer", subpath, **kwargs)

    def eval(self, text):
        tokens = self.tokenizer.encode(text)
        token_counts = Counter(tokens)
        total_tokens = len(tokens)

        # Calculate entropy
        entropy = 0
        for count in token_counts.values():
            probability = count / total_tokens
            entropy -= probability * math.log(probability, 2)
        return entropy


TokenFraction.register("token-entropy")


# ## Pattern

WORDLIST = {}

def compile_words(words: List[str], prefix="", suffix=""):
    global WORDLIST
    if isinstance(words, str):
        if "|" in words:
            words = words.split("|")
        elif "," in words:
            words = [w.strip() for w in words.split(",")]
        elif words.endswith(".txt"):
            with open(w) as f:
                ws.extend(s.strip() for s in f.readlines() if len(s.strip()) > 0)
        else:
            words = WORDLIST.get(words.replace("-", "_").lower(), [])

    ws = list(set(words))
    ws.sort()
    if prefix == "" and suffix == "":
        for w in ws:
            if "A" <= w[0] <= "z" and "A" <= w[0] <= "z":
                continue
            prefix = r"\b"
            suffix = r"\b"
            break
    pattern = "|".join(re.escape(w) for w in ws)
    if len(prefix) > 0 or len(suffix) > 0:
        return re.compile(f"{prefix}({pattern}){suffix}")
    return re.compile(pattern)


class WordCount(TextEval):
    def __init__(self, **kwargs) -> None:
        super().__init__(**(kwargs | {"_path": "word-count"}))
        subpath = self.get(kwargs, "_subpath|word_list|words|!!")
        self.pattern = compile_words(subpath)

    def eval(self, text: str) -> int:
        return len(self.pattern.findall(text))

WordCount.register("word-count|word")

class WordFraction(TextEval):
    def __init__(self, **kwargs) -> None:
        super().__init__(**(kwargs | {"_path": "word-fraction"}))
        subpath = self.get(kwargs, "_subpath|word_list|words|!!")
        self.pattern = compile_words(subpath)

    def eval(self, text: str) -> int:
        a = len(self.pattern.findall(text))
        b = len(text)
        return a / b if b != 0 else 1


WordFraction.register("word-fraction")




class ModelLoss(TextEval):
    def __init__(self, **kwargs) -> None:
        super().__init__(**(kwargs | {"_path": "model-loss"}))
        subpath = self.get(kwargs, "_subpath|model_path|model|!!")
        self.model = adhoc.load("model", subpath, **kwargs)

    def eval(self, text: str) -> int:
        return self.model.compute_loss(text)


ModelLoss.register("model-loss|loss")


class Perplexity(TextEval):
    def __init__(self, **kwargs) -> None:
        super().__init__(**(kwargs | {"_path": "perplexity"}))
        subpath = self.get(kwargs, "_subpath|model_path|model|!!")
        self.model = adhoc.load("model", subpath, **kwargs)

    def eval(self, text: str) -> int:
        return math.exp(self.model.compute_loss(text))

Perplexity.register("perplexity|ppl")

