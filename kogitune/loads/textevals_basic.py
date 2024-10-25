from .commons import *
from .textevals_ import TextEval

import re
import zlib
from collections import Counter
import math

class ByteCount(TextEval):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.get(kwargs, "encoding|=utf-8", "_unique|unique")

    def eval(self, text: str) -> int:
        if self.unique:
            return len(set(text.encode(self.encoding, errors="ignore")))
        return len(text.encode(self.encoding, errors="ignore"))

ByteCount.register("byte_length")

class ByteFraction(ByteCount):

    def eval(self, text: str) -> int:
        return super().eval(text) / len(text)

ByteFraction.register("byte_fraction")

class ZlibCount(TextEval):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.get(kwargs, "encoding|=utf-8")

    def eval(self, text: str) -> int:
        encoded = text.encode(self.encoding, errors="ignore")
        compressed = zlib.compress(encoded, level=9)
        return len(compressed)

ZlibCount.register("zlib_length")

class ZlibFraction(ZlibCount):

    def eval(self, text: str) -> int:
        return super().eval(text) / len(text)

ZlibFraction.register("zlib_fraction")


class TokenCount(TextEval):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.load("tokenizer", "_subpath|tokenizer_path|tokenizer|!!", **kwargs)

    def eval(self, text: str) -> int:
        return self.tokenizer.count(text)

TokenCount.register("token_count")


class TokenFraction(TokenCount):

    def eval(self, text: str) -> int:
        return super().eval(text) / len(text)

TokenFraction.register("token_fraction")

class TokenEntropy(TextEval):
    """
    任意のトークンリストのエントロピーを計算でき、それによりトークンの分布がどの程度多様か、
    またはどの程度予測可能かが分かります。
    エントロピーが高いほど、トークンの分布は多様で予測が難しいと言えます。
    逆にエントロピーが低い場合、トークンの分布は比較的均一で予測が容易です。
    :param tokenizer:
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.load("tokenizer", "_subpath|tokenizer_path|tokenizer|!!", **kwargs)

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

TokenEntropy.register("token_entropy")

class ModelLoss(TextEval):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.load("model", "_subpath|model_path|model|!!", **kwargs)

    def eval(self, text: str) -> int:
        return self.model.compute_loss(text)

ModelLoss.register("model_loss")

class Perplexity(TextEval):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.load("model", "_subpath|model_path|model|!!", **kwargs)

    def eval(self, text: str) -> int:
        return math.exp(self.model.compute_loss(text))

Perplexity.register("perplexity|ppl")

