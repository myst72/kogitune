from typing import Optional, List, Union, Any
import re
import zlib
from collections import Counter
import math

from .commons import *

TEXTEVAL_MAP = {}


class TextEval(adhoc.LoaderObject):
    def __init__(self, name: str, subpath: str):
        self.path = name
        self.subpath = subpath
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


TextEval.register("text-length|char-length|char")

# TODO: 未使用
# class FractionEval(adhoc.LoaderObject):
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


class TextEvalLoader(adhoc.AdhocLoader):

    def load(self, path, tag, kwargs):
        global TEXTEVAL_MAP
        path, _, subpath = path.partition(":")
        if "." in path:
            func = adhoc.load_class(path)
            if not issubclass(func, TextEval):
                raise TypeError(f"{path} is not a subclass of TextEval")
            return func(subpath, kwargs)
        path = path.lower().replace("_", "-")
        if path in TEXTEVAL_MAP:
            texteval = TEXTEVAL_MAP[path](subpath, kwargs)
        else:
            raise KeyError(path)
        # if "fraction" in kwargs:
        #     path, tag, kwargs = adhoc.parse_path(kwargs.pop("fraction"))
        #     fraction = self.load(path, tag, **kwargs)
        #     return FractionEval(texteval, fraction)
        return texteval


TextEvalLoader().register("texteval")


@adhoc.cli
def texteval_cli(**kwargs):
    """
    テキスト評価

    - files（必須）: JSONLファイル
    - texteval（必須）: 評価関数 例. "alpha-fraction"
    - input_key='text': 評価の対象となるキー
    - transform, columns: JSONを変形したいとき
    - head: 先頭だけをテストいたいとき
    - output_file: 出力先を指定したいとき
    - overwrite=False: 上書き

    例:
    ```python
    from kogitune.loads.texteval import texteval_cli

    texteval_cli(
        texteval='alpha-fraction',
        files=['jhumaneval.jsonl'],
        input_key='prompt',
        head=5,
    )
    ```
    """
    with adhoc.aargs_from(**kwargs) as aargs:
        files = listfy(aargs["files|!!"])
        texteval = adhoc.load("texteval", aargs["texteval|!!"], **kwargs)
        transform = adhoc.Transform.from_aargs(aargs)
        format_key = aargs["input_key|target|format|=text"]
        record_key = texteval.record_key()
        record_key = aargs[f"output_key|record_key|={record_key}"]

        head = aargs["head"]
        output_file = aargs[f"output_file"]
        overwrite = aargs["overwrite|=False"]
        if overwrite == False and output_file is None:
            adhoc.notice(
                "ファイル出力するには、output_fileかoverwrite=Trueを指定しよう"
            )
            adhoc.print("とりあえず、head=3だけ表示しておくね。", face="🐼")
            head = 3
        if output_file and os.path.exists(output_file):
            os.unlink(output_file)
        for filepath in files:
            record = adhoc.load("record", filepath)
            try:
                for sample in record.samples(0, head):
                    sample = transform.transform(sample)
                    text = adhoc.get_formatted_text(sample, format_key)
                    sample[record_key] = texteval(text)
                    if head:
                        adhoc.print(adhoc.dump(sample), face="🐼")
            except KeyError as e:
                report_KeyError(e, sample)
            if head is None:
                if output_file:
                    record.save(output_file, mode="a")
                else:
                    record.save()


class ByteCount(TextEval):
    def __init__(self, subpath, kwargs) -> None:
        super().__init__("byte-length", subpath)
        self.get(kwargs, "encoding|=utf-8")

    def eval(self, text: str) -> int:
        return len(text.encode(self.encoding, errors="ignore"))


ByteCount.register("byte-length|byte-count|byte")


class UniqueByteCount(TextEval):
    def __init__(self, subpath, kwargs) -> None:
        super().__init__("unique-byte-length", subpath)
        self.get(kwargs, "encoding|=utf-8")

    def eval(self, text: str) -> int:
        return len(set(text.encode(self.encoding, errors="ignore")))


UniqueByteCount.register("unique-byte-length|unique-byte-count|unique-byte")


class ByteFraction(TextEval):
    def __init__(self, subpath, kwargs) -> None:
        super().__init__("byte-fraction", subpath)
        self.get(kwargs, "encoding|=utf-8")

    def eval(self, text: str) -> int:
        a = len(text.encode(self.encoding, errors="ignore"))
        b = len(text)
        return a / b if b != 0 else 1


ByteFraction.register("byte-fraction")


class ZlibCount(TextEval):
    def __init__(self, subpath, kwargs) -> None:
        super().__init__("zlib", subpath)
        self.get(kwargs, "encoding|=utf-8")

    def eval(self, text: str) -> int:
        encoded = text.encode(self.encoding, errors="ignore")
        compressed = zlib.compress(encoded, level=9)
        return len(compressed)


ZlibCount.register("zlib-length|zlib-count|zlib")


class ZlibFraction(TextEval):
    def __init__(self, subpath, kwargs) -> None:
        super().__init__("zlib-fraction", subpath)
        self.get(kwargs, "encoding|=utf-8")

    def eval(self, text: str) -> int:
        encoded = text.encode(self.encoding, errors="ignore")
        compressed = zlib.compress(encoded, level=9)
        b = len(text)
        return len(compressed) / b if b != 0 else 1


ZlibFraction.register("zlib-fraction")


class AlphaCount(TextEval):
    def __init__(self, subpath, kwargs) -> None:
        super().__init__("alpha-count", subpath)
        self.get(kwargs, "regex|=[A-z]")
        self.pattern = re.compile(self.regex)

    def eval(self, text: str) -> int:
        return len(self.pattern.findall(text))


AlphaCount.register("alpha-count|alpha")


class AlphaFraction(AlphaCount):
    def __init__(self, subpath, kwargs) -> None:
        super().__init__("alpha-fraction", subpath)
        self.get(kwargs, "regex|=[A-z]")
        self.pattern = re.compile(self.regex)

    def eval(self, text: str) -> int:
        a = len(self.pattern.findall(text))
        b = len(text)
        return a / b if b != 0 else 1


AlphaFraction.register("alpha-fraction")


class TokenCount(TextEval):
    def __init__(self, subpath, kwargs) -> None:
        super().__init__("token", subpath)
        if subpath == "":
            subpath = self.get(kwargs, "tokenizer_path|tokenizer|!!")
        self.tokenizer = adhoc.load("tokenizer", subpath, **kwargs)

    def eval(self, text: str) -> int:
        return self.tokenizer.count(text)


TokenCount.register("token|token-count")


class TokenFraction(TextEval):
    def __init__(self, subpath, kwargs) -> None:
        super().__init__("token-fraction", subpath)
        if subpath == "":
            subpath = self.get(kwargs, "tokenizer_path|tokenizer|!!")
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

    def __init__(self, subpath, kwargs):
        """
        トークンナイザーによるエントロピー評価関数を作る
        :param tokenizer: トークンナイザー(もしくはトークンナイザー名)
        """
        super().__init__("token-entropy", subpath)
        if subpath == "":
            subpath = self.get(kwargs, "tokenizer_path|tokenizer|!!")
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
    def __init__(self, subpath, kwargs) -> None:
        super().__init__("word-count", subpath)
        if subpath == "":
            subpath = self.get(kwargs, "words|word_list|word_path")
        self.pattern = compile_words(subpath)

    def eval(self, text: str) -> int:
        return len(self.pattern.findall(text))


WordCount.register("word-count|word")


class WordFraction(TextEval):
    def __init__(self, subpath, kwargs) -> None:
        super().__init__("word-fraction", subpath)
        if subpath == "":
            subpath = self.get(kwargs, "words|word_list|word_path")
        self.pattern = compile_words(subpath)

    def eval(self, text: str) -> int:
        a = len(self.pattern.findall(text))
        b = len(text)
        return a / b if b != 0 else 1


WordFraction.register("word-fraction")


class ModelLoss(TextEval):
    def __init__(self, subpath, kwargs) -> None:
        super().__init__("model-loss", subpath)
        if subpath == "":
            subpath = self.get(kwargs, "model_path|model|!!")
        self.model = adhoc.load("model", subpath)

    def eval(self, text: str) -> int:
        return self.model.compute_loss(text)


ModelLoss.register("model-loss|loss")
