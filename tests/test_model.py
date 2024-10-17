import kogitune
import os

def test_model():
    model = kogitune.load('model', 'kkuramitsu/chico-0.03b')
    print(model)
    assert isinstance(model.generate('こんにちは', max_new_tokens=40), str)

def test_model_do_sample():
    model = kogitune.load('model', 'kkuramitsu/chico-0.03b')
    print(model)
    assert isinstance(model.generate('こんにちは', max_new_tokens=40, do_sample=True, n=2), list)
