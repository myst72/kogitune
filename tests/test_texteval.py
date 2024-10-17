import kogitune

def test_texteval_alpha_fraction():
    texteval = kogitune.load('texteval', "alpha-fraction")
    assert texteval("hogohoge") == 1.0


def test_texteval_token_fraction():
    texteval = kogitune.load('texteval', "token-fraction:llm-jp/llm-jp-3-1.8b")
    assert texteval("hogohoge") == 0.5


def test_texteval_loss():
    texteval = kogitune.load('texteval', "loss:kkuramitsu/chico-0.03b")
    assert isinstance(texteval("hogohoge"),float)
