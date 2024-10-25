from .commons import *
from .patterns_ import Pattern, compile_pattern
from .textevals_ import TextEval
import re


class CharCount(TextEval):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.get(kwargs, "_unique|unique", '_subpath|charclass|!!')
        if not self.charclass.startswith('['):
            self.pathargs['charclass'] = f'[{self.charclass}]'
        self.pattern = re.compile(self.pathargs['charclass'])

    def eval(self, text: str) -> int:
        if self.unique:
            return len(set(self.pattern.findall(text)))
        return len(self.pattern.findall(text))

CharCount.register("charactor_count|char_count")

class CharFraction(CharCount):

    def eval(self, text: str) -> int:
        return super().eval(text) / len(text)

CharFraction.register("charactor_fraction|char_fraction")


class AlphaCount(CharCount):
    def __init__(self, **kwargs) -> None:
        super().__init__(**(kwargs| {'_subpath': '[A-Za-z]'}))

AlphaCount.register("alpha_count")

class AlphaFraction(CharFraction):

    def __init__(self, **kwargs) -> None:
        super().__init__(**(kwargs| {'_subpath': '[A-Za-z]'}))

AlphaFraction.register("alpha_fraction")

class AlnumCount(CharCount):
    def __init__(self, **kwargs) -> None:
        super().__init__(**(kwargs| {'_subpath': '[A-Za-z0-9]'}))

AlnumCount.register("alnum_count")

class AlnumFraction(CharFraction):

    def __init__(self, **kwargs) -> None:
        super().__init__(**(kwargs| {'_subpath': '[A-Za-z0-9]'}))

AlnumFraction.register("alnum_fraction")

## Pattern

class PatternCount(TextEval):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.get(kwargs, "_unique|unique")
        self.load('pattern', "_subpath|pattern|!!", **kwargs)

    def eval(self, text: str) -> int:
        if self.unique:
            return self.pattern.unique_count(text)
        return self.pattern.count(text)

PatternCount.register("pattern_count")

class PatternFraction(PatternCount):

    def eval(self, text: str) -> int:
        return super().eval(text) / len(text)

PatternFraction.register("pattern_fraction")

## Pattern

class WordCount(PatternCount):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.get(kwargs, "_subpath|word_list|words|!!", "_unique|unique")
        words = adhoc.get_words(kwargs, "_subpath|word_list|words|!!")
        compiled = compile_pattern({
            'words': words
        })
        self.pattern = Pattern(_compiled=compiled)

WordCount.register("word_count")

class WordFraction(WordCount):

    def eval(self, text: str) -> int:
        return super().eval(text) / len(text)

WordFraction.register("word_fraction")



## Extractor 

class ExtractorLength(TextEval):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.load('extractor', '_subpath|extractor|!!', **kwargs)

    def eval(self, text: str):
        length = sum(len(d) for d in self.extractor.extract(text))
        return length

ExtractorLength.register("extract_length")

class ExtractorFraction(ExtractorLength):

    def eval(self, text: str) -> int:
        return super().eval(text) / len(text)

ExtractorFraction.register("extract_fraction")
