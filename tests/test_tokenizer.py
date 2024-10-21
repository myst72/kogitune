import kogitune
import os
import tempfile

def test_tokenizer():
    tokenizer = kogitune.load('tokenizer', "llm-jp/llm-jp-3-1.8b")
    assert(isinstance(tokenizer('こんにちは世界, hello world'), list))

def test_tokenizer_simple():
    tokenizer = kogitune.load('tokenizer:simple')
    assert(isinstance(tokenizer('こんにちは世界, hello world'), list))

def test_tokenizer_sudachipy():
    tokenizer = kogitune.load('tokenizer:sudachipy')
    assert(isinstance(tokenizer('こんにちは世界, hello world'), list))

def test_tokenizer_janome():
    tokenizer = kogitune.load('tokenizer:janome')
    assert(isinstance(tokenizer('こんにちは世界, hello world'), list))



def test_add_vocab():
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        kogitune.cli.add_vocab( 
                    tokenizer_path='PY007/TinyLlama-1.1B-Chat-v0.1',
                    words='/Users/kimio/Downloads/swallow_addvocab.jsonl',
                    word_key='Token',
                    multi_index=['<0x00>', '<0x01>'],
        )

def test_train_bpe():
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        kogitune.cli.train
        kogitune.load('cli', 'add_vocab', 
                    #tokenizer_path='llm-jp/llm-jp-3-1.8b',
                    tokenizer_path='PY007/TinyLlama-1.1B-Chat-v0.1',
                    words='/Users/kimio/Downloads/swallow_addvocab.jsonl',
                    word_key='Token',
                    multi_index=['<0x00>', '<0x01>'],
        )