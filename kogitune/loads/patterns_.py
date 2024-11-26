from typing import Optional, List, Union, Any
import re

from .commons import *

PATTERN_MAP = {}

regex = adhoc.safe_import('regex')

regex_operators = re.compile(r'[.\*\+\?\[\]\(\)\|\^\$]')

class PatternLoader(adhoc.AdhocLoader):

    def load_modules(self, path, kwargs):
        from .patterns_chico import pattern_config_commons
        from .patterns_langs import pattern_config_lang
    
    def load_default(self, path, kwargs):
        ## まずパターンライブラリを調べる
        compiled = find_compiled_pattern(path)
        if compiled:
            kwargs['_compiled'] = compiled
            return Pattern(**kwargs)
        if regex_operators.search(path):
            # 正規表現の演算子が含まれれば正規表現とみなす。
            kwargs['_pattern'] = path
            return Pattern(**kwargs)
        return super().load_default(path, kwargs)

PatternLoader(PATTERN_MAP).register("re|pattern")

class Pattern(adhoc.AdhocObject):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scheme = 'pattern'
        if '_compiled' in kwargs:
            self.compiled = kwargs['_compiled']
        elif '_pattern' in kwargs:
            self.compiled = self.compile(kwargs['_pattern'])

    def __call__(self, text: str) -> str:
        return self.replace(text)

    def compile(self, *patterns: List[str], flags=0):
        return re.compile("|".join(patterns), flags=flags)

    def search(self, text:str):
        return self.compiled.search(text)

    def contains(self, text:str) -> bool:
        return bool(self.compiled.search(text))

    def findall(self, text: str) -> str:
        return self.compiled.findall(text)

    def count(self, text: str) -> int:
        return len(self.compiled.findall(text))

    def unique_count(self, text: str) -> int:
        return len(set(self.compiled.findall(text)))

    def count_match(self, text: str) -> int:
        before = text.count("💣")
        replaced = self.pattern.sub(text, "💣")
        return replaced.count("💣") - before

    def extract(self, text: str) -> List[str]:
        """
        Extractor と同じインターフェース
        """
        matched = self.findall(text)
        ss = []
        for m in matched:
            if isinstance(m, str):
                ss.append(m)
            else:
                ss.append('\n'.join(m))
        if len(ss) == 0:
            ss.append('') # ss[0] = '' は保証する
        return ss

    def replace(self, text: str, replaced_text:str='<MATCH>') -> str:
        return self.compiled.sub(replaced_text, text)


    def registor(self, names: str):
        global PATTERN_MAP
        for name in adhoc.list_keys(names):
            PATTERN_MAP[name.lower()] = self.__class__


## PatternDB

def RE(*patterns: List[str], flags=0):
    return regex.compile("|".join(patterns), flags=flags)

PATTERN_DATABASE = {}

def register_pattern(pattern_config_map: dict):
    global PATTERN_DATABASE
    for names, pattern_config in pattern_config_map.items():
        for name in adhoc.list_keys(names):
            # if name in PATTERN_DATABASE:
            #     adhoc.verbose_print(f'Duplicated registration {name}', dump=pattern_config)
            PATTERN_DATABASE[name] = pattern_config

def find_pattern(key):
    global PATTERN_DATABASE
    if is_config(key):
        return load_config(key)
    if key in PATTERN_DATABASE:
        return PATTERN_DATABASE[key]
    simkey = adhoc.find_simkey(PATTERN_DATABASE, key, max_distance=1)
    if simkey:
        adhoc.verbose_print(f'Typo? {key} => {simkey}')
        return PATTERN_DATABASE[simkey]
    return None

def find_compiled_pattern(key):
    pattern_config = find_pattern(key)
    if pattern_config:
        if 'compiled' in pattern_config:
            return pattern_config['compiled']
        compiled = compile_pattern(pattern_config)
        pattern_config['compiled'] = compiled
        return compiled
    return None

def compile_pattern(config: dict):
    if 'patterns' in config:
        return RE(*config['patterns'], flags=config.get('flags', 0))
    if 'words' in config:
        words = set(config['words'])
        if config.get('capitalize', False):
            for w in config['words']:
                words.add(w.capitalize())
        pattern = Trie(words).pattern()
        if config.get('word_segmentation', False):
            pattern = f'\\b{pattern}\\b'
        return regex.compile(pattern, flags=config.get('flags', 0))
    adhoc.verbose_print('パターンが作れません', dump=config)
    return None

def test_pattern(key):
    pattern_config = find_pattern(key)
    if pattern_config:
        pattern = adhoc.load('pattern', key)
        if 'tests' in pattern_config:
            for testcase, _ in pattern_config['tests']:
                print(key, testcase)
                print(" =>", pattern.extract(testcase))
                print(" =>", pattern.replace(testcase))

## Trie

#author:         rex
#blog:           http://iregex.org
#filename        trie.py
#created:        2010-08-01 20:24
#source uri:     http://iregex.org/blog/trie-in-python.html

# escape bug fix by fcicq @ 2012.8.19
# python3 compatible by EricDuminil @ 2017.03.


class Trie():
    """Regexp::Trie in python. Creates a Trie out of a list of words. The trie can be exported to a Regexp pattern.
    The corresponding Regexp should match much faster than a simple Regexp union."""

    def __init__(self, words=None):
        self.data = {}
        if words:
            for w in words:
                self.add(w)

    def add(self, word):
        ref = self.data
        for char in word:
            ref[char] = char in ref and ref[char] or {}
            ref = ref[char]
        ref[''] = 1

    def dump(self):
        return self.data

    def quote(self, char):
        return re.escape(char)

    def _pattern(self, pData):
        data = pData
        if "" in data and len(data.keys()) == 1:
            return None

        alt = []
        cc = []
        q = 0
        for char in sorted(data.keys()):
            if isinstance(data[char], dict):
                try:
                    recurse = self._pattern(data[char])
                    alt.append(self.quote(char) + recurse)
                except:
                    cc.append(self.quote(char))
            else:
                q = 1
        cconly = not len(alt) > 0

        if len(cc) > 0:
            if len(cc) == 1:
                alt.append(cc[0])
            else:
                alt.append('[' + ''.join(cc) + ']')

        if len(alt) == 1:
            result = alt[0]
        else:
            result = "(?:" + "|".join(alt) + ")"

        if q:
            if cconly:
                result += "?"
            else:
                result = "(?:%s)?" % result
        return result

    def pattern(self):
        return self._pattern(self.dump())


### 日本語関連

pattern_config_ja = {
    "hiragana|hira": {
        "patterns": [r'[ぁ-ん]'],
        "flag": 0,
    },
    "katakana|kata": {
        "patterns": [r'[ァ-ヶー・]'],
        "flag": 0,
    },
    "hirakata|ja": {
        "patterns": [r'[ぁ-んァ-ヶー・。、]'],
        "flag": 0,
    },
}
register_pattern(pattern_config_ja)

pattern_hirakata = None

def contains_japanese(text: str) -> bool:
    """
    テキストに日本語を含むかどうかを判定する

    :param text: 判定するテキスト
    :return: 日本語を含む場合はTrue、そうでない場合はFalse
    """
    if pattern_hirakata is None:
        pattern_hirakata = adhoc.load('pattern', 'hirakata')
    return pattern_hirakata.contains(text)

##  Extractor


EXTRACTOR_MAP = {}

class ExtractorLoader(adhoc.AdhocLoader):

    def load_modules(self, path, kwargs):
        from .extractors_base import LinesExtractor
        from .extractors_py import PythonSyntacticExtractor

    def load_default(self, path, kwargs):
        if regex_operators.search(path):
            # 正規表現の演算子が含まれれば正規表現とみなす。
            kwargs['_pattern'] = path
            return Pattern(**kwargs)
        return super().load_default(path, kwargs)

ExtractorLoader(EXTRACTOR_MAP).register("extractor")

class Extractor(adhoc.AdhocObject):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scheme = 'extractor'

    def extract(self, text: str) -> List[str]:
        return [text]

    @classmethod
    def register(cls, names: str):
        global EXTRACTOR_MAP
        for name in adhoc.list_keys(names):
            EXTRACTOR_MAP[name.lower()] = cls

Extractor.register('none')

class StopWordExtractor(Extractor):
    """
    テキストからストップワードまでを取り出すExtractor
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.load('pattern', '_subpath|pattern|stop_words|!!', **kwargs)

    def extract(self, text):
        matched = self.pattern.search(text)
        if matched:
            text = text[: matched.start()]
        return [text]

StopWordExtractor.register('stop_words|stop_word')

stopword_pattern_config = {
    "wikipedia_footnote_ja": {
        "words": [
            "脚注",
            "関連項目",
            "日本国内の関連項目",
            "出典",
            "出典・脚注",
            "参照",
            "外部リンク",
            "参考文献",
            "その他関連事項",
            "Footnotes",
            "See also",
            "Further reading",
            "Bibliography",
            "References",
            "Notes",
            "Citations",
            "Sources",
            "External links",
        ],
    }
}

register_pattern(stopword_pattern_config)

