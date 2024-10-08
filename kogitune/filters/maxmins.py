from typing import Optional, List, Union, Any
import re
import zlib
from collections import Counter
import math

import kogitune.adhocs as adhoc
from .filters import TextFilter

class TextEval(object):
    def __init__(self, kwargs):
        self.rec = {}
        # fraction_fn が設定されていると分母にして比率を計算する
        self.fraction_fn = None
        self.reverse = False

    def setups(self, kwargs, *keys):
        adhoc.kwargs_from_stacked(**kwargs).record(
            *keys,
            dic=self.rec, field=self,
        )

    def setup_fraction(self, kwargs):
        self.setups(kwargs, 
                    'fraction|=text-length', f'reverse|={self.reverse}')
        self.fraction_fn = load_eval_fn(self.fraction)

    def eval(self, text: str):
        return 0

    def __call__(self, text: str):
        if self.fraction_fn is None:
            return self.eval(text)
        if self.reverse:
            count = self.eval(text)
            if count != 0:
                return self.fraction_fn(text) / count
        else:
            count = self.fraction_fn(text)
            if count != 0:
                return self.eval(text) / count
        return 0

    def __repr__(self):
        #return json.dumps(self.rec, indent=2)
        return f'{self.rec} {self.__class__.__name__}'


class OneCount(TextEval):
    def __init__(self, kwargs) -> None:
        super().__init__(kwargs)

    def eval(self, text:str) -> int:
        return 1

class CharCount(TextEval):
    def __init__(self, kwargs) -> None:
        super().__init__(kwargs)
        self.setups(kwargs, 'unique|=False')

    def eval(self, text:str) -> int:
        if self.unique:
            return len(set(list(text)))
        return len(text)

class ByteCount(TextEval):
    def __init__(self, kwargs) -> None:
        super().__init__(kwargs)
        self.setups(kwargs, 'encoding|=utf-8', 'unique|=False')

    def eval(self, text:str) -> int:
        if self.unique:
            return len(set(text.encode(self.encoding, errors='ignore')))
        return len(text.encode(self.encoding, errors='ignore'))

class ByteFraction(ByteCount):
    def __init__(self, kwargs) -> None:
        super().__init__(kwargs)
        self.setup_fraction(kwargs)

class ZlibFraction(TextEval):
    def __init__(self, kwargs) -> None:
        super().__init__(kwargs)
        self.setup_fraction(kwargs)

    def eval(self, text:str) -> int:
        encoded = text.encode("utf-8", errors='ignore')
        compressed = zlib.compress(encoded, level=9)    
        return len(compressed)

class AlphaCount(TextEval):
    def __init__(self, kwargs) -> None:
        super().__init__(kwargs)
        self.setups(kwargs, 'regex|=[A-z]', 'unique|=False')
        self.pattern = re.compile(self.regex)

    def eval(self, text:str) -> int:
        if self.unique:
            return len(set(self.pattern.findall(text)))
        return len(self.pattern.findall(text))


class AlphaFraction(AlphaCount):
    def __init__(self, kwargs) -> None:
        super().__init__(kwargs)
        self.setup_fraction(kwargs)

class TokenCount(TextEval):
    def __init__(self, kwargs) -> None:
        super().__init__(kwargs)
        self.setups(kwargs, 'tokenizer_path|tokenizer|!!', 'unique|=False')
        self.tokenizer = adhoc.load_tokenizer(tokenizer=self.tokenizer_path)

    def eval(self, text:str) -> int:
        if self.unique:
            return len(set(self.tokenizer.encode(text)))
        return len(self.tokenizer.encode(text))

class TokenFraction(TokenCount):
    def __init__(self, kwargs) -> None:
        super().__init__(kwargs)
        self.setup_fraction(kwargs)

class TokenEntropy(TextEval):
    """
    任意のトークンリストのエントロピーを計算でき、それによりトークンの分布がどの程度多様か、
    またはどの程度予測可能かが分かります。
    エントロピーが高いほど、トークンの分布は多様で予測が難しいと言えます。
    逆にエントロピーが低い場合、トークンの分布は比較的均一で予測が容易です。
    :param tokenizer:
    """

    def __init__(self, kwargs):
        """
        トークンナイザーによるエントロピー評価関数を作る
        :param tokenizer: トークンナイザー(もしくはトークンナイザー名)   
        """
        super().__init__(kwargs)
        self.setups(kwargs, 'tokenizer_path|tokenizer|!!')
        self.tokenizer = adhoc.load_tokenizer(tokenizer=self.tokenizer_path)

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


alpha_pattern = re.compile(r'[A-z]')

def alpha_fraction(text: str) -> float:
    """
    英文字の比率を算出する
    """
    count = len(text)
    if count > 0:
        return len(alpha_pattern.findall(text)) / count
    return 0.0 

## Pattern

from .patterns import compile_words
from .utils_en import common_english_words_pattern

class PatternEval(TextEval):
    def __init__(self, pattern, kwargs) -> None:
        super().__init__(kwargs)
        self.setups(kwargs, 'unique|=False')
        self.pattern = pattern

    def eval(self, text:str) -> int:
        if self.unique:
            return len(set(self.pattern.findall(text)))
        return len(self.pattern.findall(text))

class EnglishWordCount(PatternEval):
    """
    与えられたテキストに英単語(欧文単語)が含まれるか判定する評価関数
    """
    def __init__(self, kwargs):
        """
        与えられたテキストに英単語が含まれるか判定する評価関数を作る
        :param words: 英単語のリスト(省略した場合は GPT-4による頻出英単語)
        """
        words = kwargs.pop('words')
        if words:
            pattern = compile_words(words, prefix=r'\b', suffix=r'\b')
        else:
            pattern = common_english_words_pattern
        super().__init__(pattern, kwargs)

class EnglishWordFraction(PatternEval):
    """
    与えられたテキストに英単語(欧文単語)が含まれるか判定する評価関数
    """
    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.setup_fraction(kwargs)


EVAL_MAP = {
    'none': OneCount, 
    'text-length': CharCount,
    'byte': ByteCount,
    'byte-fraction': ByteFraction,
    'zlib-fraction': ZlibFraction,
    'alpha': AlphaCount,
    'alpha-fraction': AlphaFraction,
    'token': TokenCount,
    'token-fraction': TokenFraction,
    'token-entropy': TokenEntropy,
    'english': EnglishWordCount,
    'english-fraction': EnglishWordFraction,
    # 'word-ja': JapaneseWordCounter,
}

def load_eval_fn(eval, parent: dict = None):
    if isinstance(eval, TextEval):
        return eval
    path, kwargs = adhoc.parse_path_args(eval, parent_args=parent)
    if path.startswith('tokenizer:'):
        kwargs['tokenizer_path'] = path[10:]
        return TokenCount(kwargs)
    if path.startswith('regex:'):
        kwargs['regex'] = path[6:]
        return AlphaCount(kwargs)
    if '.' in path:
        func = adhoc.load_class(path)
        if not issubclass(func, TextEval):
            raise TypeError(f'{path} is not a subclass of TextEval')
        return func(kwargs)
    if path in EVAL_MAP:
        func = EVAL_MAP[path]
        return func(kwargs)
    else:
        adhoc.warn(unknown_eval=eval, expected=list(EVAL_MAP.keys()))


DEFAULT_PERCENTILES = [0.01, 0.05, 0.1, 0.2, 0.25, 0.33, 0.5, 0.66, 0.75, 0.8, 0.9, 0.95, 0.99]

class MaxMinFilter(TextFilter):
    """
    評価関数の最大値と最小値からフィルターする
    """
    def __init__(self, *args, **kwargs):
        """
        評価関数フィルタを作る
        """
        super().__init__(*args, **kwargs)
        aargs = adhoc.kwargs_from_stacked(**kwargs)
        #print('@@@', aargs)
        aargs.record(
            'eval|!!',
            'max_inclusive|max',
            'min_inclusive|min',
            'record_key',
            'sample|head|N|=0',
            '_prefix|prefix|=',
            field=self, dic=self.rec,
        )
        self.eval_fn = load_eval_fn(self.eval, parent=aargs)
        self.rec = self.rec | self.eval_fn.rec
        if self.record_key == True:
            self.record_key = self.eval
        if self.max_inclusive == None and self.min_inclusive == None:
            if self.sample == 0:
                self.sample = 100000
        if self.sample > 0:
            self.samples = []

    def __call__(self, text: str, record: dict) -> Optional[str]:
        value = self.eval_fn(text)
        if self.record_key:
            record[self.record_key] = round(value,5)
        if self.sample > 0:
            self.samples.append(value)
            if len(self.samples) == self.sample:
                self.describe()
        if (self.min_inclusive and self.min_inclusive > value):
            adhoc.print(f'[DROP] {value} < min={self.min_inclusive} {repr(text)}', watch=self.record_key)
            return None
        if (self.max_inclusive and self.max_inclusive < value):
            adhoc.print(f'[DROP] {value} > max={self.max_inclusive} {repr(text)}', watch=self.record_key)
            return None
        return text

    def describe(self):
        import pandas as pd
        if self.sample > 0 and len(self.samples) > 0:
            self.sample = 0
            name = self.record_key or self.eval
            df = pd.DataFrame({name: self.samples})
            adhoc.print(df.describe(percentiles=DEFAULT_PERCENTILES), face='')
            df.to_csv(f'{self.prefix}{name}.csv', index=None)
            adhoc.saved(f'{self.prefix}{name}.csv', f'分布データ({self.eval})')
            self.samples = []
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns
                sns.displot(df, stat='probability')
                plt.savefig(f'{self.prefix}{name}')
                adhoc.saved(f'{self.prefix}{name}.png', f'ヒストグラム画像({self.eval})')
                plt.clf()
            except:
                pass

def maxmin(eval_fn_name, **kwargs):
    try:
        return MaxMinFilter(eval=eval_fn_name, **kwargs)
    except KeyError as e:
        print(e)
        raise e

def filter_maxmin_cli(**kwargs):
    with adhoc.kwargs_from_stacked(**kwargs) as aargs:
        if 'record_key' not in aargs:
            text_filter = MaxMinFilter(record_key=True, **kwargs)
        else:
            text_filter = MaxMinFilter(**kwargs)
        text_filter.run_for_cli(**kwargs)
