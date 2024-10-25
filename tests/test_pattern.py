import kogitune
from kogitune.loads import Pattern

def test_pattern():
    patern = kogitune.load('pattern', "url")
    replaced = patern.replace("http://live.nicovideo.jp/watch/lv265893495 #nicoch2585696")
    assert replaced == '<MATCH> #nicoch2585696'

def test_pattern_contains():
    patern = kogitune.load('pattern', "ja")
    assert patern.contains("おはよう") == True
    assert patern.contains("#nicoch2585696") == False

def test_pattern_count():
    patern = kogitune.load('pattern', "hiragana")
    assert patern.count("おはよう") == 4

def test_pattern_extract():
    patern = kogitune.load('pattern', r"\b[ABCD]\b")
    assert patern.extract("Answer (A)") == ['A']

WIKIPEDIA_TEST_DOC = """hoge
hoge
出典
hoge
"""

def test_extractor_stopword():
    f = kogitune.load('extractor', "stop_word:wikipedia_footnote_ja")
    print(f.encode_as_json())
    text = f.extract(WIKIPEDIA_TEST_DOC)
    print(text)
    assert '出典' not in text

def test_texteval_stopword_fraction():
    f = kogitune.load('texteval', "extracted_fraction:stop_word:wikipedia_footnote_ja")
    print(f.encode_as_json())
    v = f.eval(WIKIPEDIA_TEST_DOC)
    assert v > 0.5
