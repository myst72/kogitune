import kogitune
import os
import tempfile
import pytest

def test_tokenizer():
    tokenizer = kogitune.load('tokenizer', "llm-jp/llm-jp-3-1.8b")
    assert(isinstance(tokenizer('こんにちは世界, hello world'), list))

def test_tokenizer_simple():
    tokenizer = kogitune.load('tokenizer:simple')
    assert(isinstance(tokenizer('こんにちは世界, hello world'), list))

def has_module(name):
    import importlib
    try:
        importlib.import_module(name)
        return True
    except ModuleNotFoundError as e:
        return False

@pytest.mark.skipif(not has_module('sudachipy'), 
                    reason="requires sudachipy")
def test_tokenizer_sudachipy():
    tokenizer = kogitune.load('tokenizer:sudachipy')
    assert(isinstance(tokenizer('こんにちは世界, hello world'), list))

@pytest.mark.skipif(not has_module('janome'), 
                    reason="requires janome")
def test_tokenizer_janome():
    tokenizer = kogitune.load('tokenizer:janome')
    assert(isinstance(tokenizer('こんにちは世界, hello world'), list))

def test_add_vocab():
    from kogitune.datasets.chunks import download_file
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        local_file = download_file('https://github.com/kkuramitsu/kogitune/raw/refs/heads/main/test_data/swallow_addvocab.jsonl', 'swallow_addvocab.jsonl')
        kogitune.cli.add_vocab( 
                    tokenizer_path='PY007/TinyLlama-1.1B-Chat-v0.1',
                    words=local_file,
                    word_key='Token',
                    multi_index=['<0x00>', '<0x01>'],
        )

def test_train_bpe():
    from kogitune.datasets.chunks import download_file, decompress_file
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        local_file = download_file('https://github.com/kkuramitsu/kogitune/raw/refs/heads/main/test_data/prog_text.txt.zst')
        decompress_file(local_file, 'prog_text.txt')
        kogitune.cli.train_bpe(
            files = ['prog_text.txt'],
            vocab_size=4000,
        )
