import kogitune

def test_pattern():
    patern = kogitune.load('pattern', "url#<URL>")
    replaced = patern.replace("http://live.nicovideo.jp/watch/lv265893495 #nicoch2585696")
    assert replaced == '<URL> #nicoch2585696'
