import kogitune

def test_texteval():
    texteval = kogitune.load("texteval:text_length")
    assert texteval.test_reload()
    assert texteval("hogohoge") == 8

def test_texteval_simkey():
    texteval = kogitune.load("texteval:text-length")
    assert texteval.test_reload()
    assert texteval("hogehoge") == 8

def test_texteval_byte_length():
    texteval = kogitune.load('texteval', "byte_length")
    assert texteval.test_reload()
    assert texteval("ABCabc123あいう") == 18

def test_texteval_byte_fraction():
    texteval = kogitune.load('texteval', "byte_fraction")
    assert texteval.test_reload()
    assert texteval("ABCabc123あいう") == 18 / 12

def test_texteval_zlib_length():
    texteval = kogitune.load('texteval', "zlib_length")
    assert texteval.test_reload()
    assert texteval("ABCabc123あいう") == 26

def test_texteval_zlib_fraction():
    texteval = kogitune.load('texteval', "zlib_fraction")
    assert texteval.test_reload()
    assert texteval("ABCabc123あいう") == 26 / 12

def test_texteval_char_count():
    texteval = kogitune.load('texteval', "char_count:[あい]")
    assert texteval.test_reload()
    assert texteval("ABCabc123あいう") == 2

def test_texteval_char_fraction():
    texteval = kogitune.load('texteval', "char_fraction:[あい]")
    assert texteval.test_reload()
    assert texteval("ABCabc123あいう") == 2 / 12

def test_texteval_alpha_count():
    texteval = kogitune.load('texteval', "alpha_count")
    assert texteval.test_reload()
    assert texteval("ABCabc123あいう") == 6

def test_texteval_unique_alpha_count():
    texteval = kogitune.load("texteval:unique_alpha_count")
    assert texteval.test_reload()
    assert texteval("hogehoge") == 4

def test_texteval_alpha_fraction():
    texteval = kogitune.load('texteval', "alpha_fraction")
    assert texteval.test_reload()
    assert texteval("ABCabc123あいう") == 0.5

def test_texteval_alnum_count():
    texteval = kogitune.load('texteval', "alnum_count")
    assert texteval.test_reload()
    assert texteval("ABCabc123あいう") == 9

def test_texteval_alpha_fraction():
    texteval = kogitune.load('texteval', "alnum_fraction")
    assert texteval.test_reload()
    assert texteval("ABCabc123あいう") == 0.75

# def test_texteval_word_count():
#     texteval = kogitune.load('texteval', "word_count:ja")
#     assert texteval("こんにちは世界, Hello World!!") == 23

# def test_texteval_word_fraction():
#     texteval = kogitune.load('texteval', "word_fraction:ja")
#     assert texteval("こんにちは世界, Hello World!!") > 1.0

def test_texteval_pattern_count():
    texteval = kogitune.load('texteval', "pattern_count:ja")
    #assert texteval.test_reload()    
    assert texteval("こんにちは世界, Hello World!!") == 1

def test_texteval_pattern_fraction():
    texteval = kogitune.load('texteval', "pattern_fraction:ja")
    #assert texteval.test_reload()
    assert texteval("こんにちは世界, Hello World!!") < 1.0

def test_texteval_regex_count():
    texteval = kogitune.load('texteval', "pattern_count:[A-Z]")
    assert texteval("こんにちは世界, Hello World!!") == 2

def test_texteval_regex_fraction():
    texteval = kogitune.load('texteval', "pattern_fraction:[A-Z]")
    assert texteval("こんにちは世界, Hello World!!") < 1.0

def test_texteval_token_count():
    texteval = kogitune.load('texteval', "token_count:llm-jp/llm-jp-3-1.8b")
    assert texteval("hogohoge") == 4

def test_texteval_token_fraction():
    texteval = kogitune.load('texteval', "token_fraction:llm-jp/llm-jp-3-1.8b")
    assert texteval("hogohoge") == 0.5

def test_texteval_token_entropy():
    texteval = kogitune.load('texteval', "token_entropy:llm-jp/llm-jp-3-1.8b")
    assert texteval("hogohoge") > 1.0

def test_texteval_loss():
    texteval = kogitune.load('texteval', "model_loss:kkuramitsu/chico-0.03b")
    assert isinstance(texteval("hogohoge"),float)

def test_texteval_perplexity():
    texteval = kogitune.load('texteval', "perplexity:kkuramitsu/chico-0.03b")
    assert isinstance(texteval("hogohoge"),float)
