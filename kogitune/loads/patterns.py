from typing import Optional, List, Union, Any
import re

from .commons import *

PATTERN_MAP = {}

class PatternLoader(adhoc.AdhocLoader):

    def load_from_map(self, path, kwargs):
        pat = super().load_from_map(path, kwargs)
        # if "fraction" in kwargs:
        #     path, tag, kwargs = adhoc.parse_path(kwargs.pop("fraction"))
        #     fraction = self.load(path, tag, **kwargs)
        #     return FractionEval(texteval, fraction)
        return pat

PatternLoader(PATTERN_MAP).register("re|pattern")

class Pattern(adhoc.AdhocObject):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.set_default_replaced("<URL>")
        self.pattern = self.compile(r"https?://[\w/:%#\$&\?~\.\,=\+\-\\_]+")

    def set_default_replaced(self, replaced):
        if self.tag == '':
            self.tag = replaced

    def __call__(self, text: str) -> str:
        return self.replace(text)

    def compile(self, *patterns: List[str], flags=0):
        return re.compile("|".join(patterns), flags=flags)

    def count_match(self, text: str) -> int:
        before = text.count("💣")
        replaced = self.pattern.sub(text, "💣")
        return replaced.count("💣") - before

    def replace(self, text: str) -> str:
        return self.pattern.sub(self.tag, text)

    def __repr__(self):
        return f"{self.path}#{self.tag}"

    def registor(self, names: str):
        global PATTERN_MAP
        for name in adhoc.list_keys(names):
            PATTERN_MAP[name.lower()] = self.__class__


# class ComposePattern(adhoc.AdhocObject):
#     def __init__(self, *Patterns):
#         self.Patterns = Patterns

#     def replace(self, text: str) -> str:
#         for re in self.Patterns:
#             text = re.replace(text)
#         return text

#     def count_match(self, text: str) -> int:
#         before = text.count("💣")
#         for re in self.Patterns:
#             text = re.replace(text, "💣")
#         return text.count("💣") - before

#     def __repr__(self):
#         return ":".join(f"{re}" for re in self.Patterns)


## URL


class patternURL(Pattern):
    """
    text 中のURLを<url>に置き換える

    >>> reURL("http://www.peugeot-approved.net/UWS/WebObjects/UWS.woa/wa/carDetail?globalKey=uwsa1_1723019f9af&currentBatch=2&searchType=1364aa4ee1d&searchFlag=true&carModel=36&globalKey=uwsa1_1723019f9af uwsa1_172febeffb0, 本体価格 3,780,000 円")
    '<url> uwsa1_172febeffb0, 本体価格 3,780,000 円'

    >>> replace_url("「INVADER GIRL!」https://www.youtube.com/watch?v=dgm6-uCDVt0")
    '「INVADER GIRL!」<url>'

    >>> replace_url("http://t.co/x0vBigH1Raシグネチャー")
    '<url>'

    >>> replace_url("(http://t.co/x0vBigH1Ra)シグネチャー")
    '(<url>)シグネチャー'

    >>> replace_url("kindleにあるなーw http://www.amazon.co.jp/s/ref=nb_sb_noss?__mk_ja_JP=%E3%82%AB%E3%82%BF%E3%82%AB%E3%83%8A&url=search-alias%3Ddigital-text&field-keywords=%E3%82%A2%E3%82%B0%E3%83%8D%E3%82%B9%E4%BB%AE%E9%9D%A2")
    'kindleにあるなーw <url>'

    >>> replace_url("http://live.nicovideo.jp/watch/lv265893495 #nicoch2585696")
    '<url> #nicoch2585696'

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_default_replaced('URL')
        self.pattern = self.compile(r"https?://[\w/:%#\$&\?~\.\,=\+\-\\_]+")  # 結構, 適当

patternURL().registor("url")
