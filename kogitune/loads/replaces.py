from typing import Optional, List, Union, Any
import re

from .commons import *

REPLACEMENT_MAP = {}


class Replacement(adhoc.LoaderObject):
    def __init__(self, path=None, replaced="", kwargs=None):
        self.path = path
        self.replaced = replaced or "<URL>"
        self.pattern = self.compile(r"https?://[\w/:%#\$&\?~\.\,=\+\-\\_]+")

    def __call__(self, text: str) -> str:
        return self.replace(text)

    def compile(self, *patterns: List[str], flags=0):
        return re.compile("|".join(patterns), flags=flags)

    def replace(self, text: str) -> str:
        return self.pattern.sub(self.replaced, text)

    def count_match(self, text: str) -> int:
        before = text.count("💣")
        replaced = self.replace(text, "💣")
        return replaced.count("💣") - before

    def __repr__(self):
        return f"{self.path}#{self.replaced}"

    def registor(self, names: str):
        global REPLACEMENT_MAP
        for name in adhoc.list_keys(names):
            REPLACEMENT_MAP[name.lower()] = self.__class__


# class ComposeReplacement(adhoc.LoaderObject):
#     def __init__(self, *replacements):
#         self.replacements = replacements

#     def replace(self, text: str) -> str:
#         for re in self.replacements:
#             text = re.replace(text)
#         return text

#     def count_match(self, text: str) -> int:
#         before = text.count("💣")
#         for re in self.replacements:
#             text = re.replace(text, "💣")
#         return text.count("💣") - before

#     def __repr__(self):
#         return ":".join(f"{re}" for re in self.replacements)


class ReplacementLoader(adhoc.AdhocLoader):

    def load(self, path, tag, kwargs):
        global REPLACEMENT_MAP
        if "." in path:
            cls = adhoc.load_class(path)
            if not issubclass(cls, Replacement):
                raise TypeError(f"{path} is not a subclass of replacement")
            return cls(path, tag, kwargs)
        if path in REPLACEMENT_MAP:
            rep = REPLACEMENT_MAP[path](path, tag, kwargs)
        else:
            raise KeyError(path)
        return rep


ReplacementLoader().register("re")

## URL


class reURL(Replacement):
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

    def __init__(self, path=None, replaced="", kwargs=None):
        self.path = path
        self.replaced = replaced or "<URL>"
        self.pattern = self.compile(r"https?://[\w/:%#\$&\?~\.\,=\+\-\\_]+")


reURL().registor("url")
