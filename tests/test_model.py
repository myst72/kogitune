import kogitune
import os
import pytest

def test_model():
    model = kogitune.load('model', 'kkuramitsu/chico-0.03b')
    print(model)
    assert isinstance(model.generate('こんにちは', max_new_tokens=40), str)

def test_model_do_sample():
    model = kogitune.load('model', 'kkuramitsu/chico-0.03b')
    print(model)
    assert isinstance(model.generate('こんにちは', max_new_tokens=40, do_sample=True, n=2), list)

@pytest.mark.skipif('OPENAI_API_KEY' not in os.environ, 
                    reason="requires OPENAI_API_KEY")
def test_openai_gpt4():
    model = kogitune.load('model', 'openai:gpt-4o-mini')
    print(model)
    assert isinstance(model.generate('こんにちは', max_new_tokens=40, n=2), list)

