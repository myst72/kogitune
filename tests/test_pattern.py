import kogitune

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
    assert patern.extract("Answer (A)") == 'A'
