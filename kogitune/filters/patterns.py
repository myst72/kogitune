from typing import List

import os
import re
import regex


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
        return regex.compile(f"{prefix}({pattern}){suffix}")
    return regex.compile(pattern)


def RE(*patterns: List[str], flags=0):
    return regex.compile("|".join(patterns), flags=flags)


def replace_pattern(pattern, text, replaced):
    return pattern.sub(replaced, text)
