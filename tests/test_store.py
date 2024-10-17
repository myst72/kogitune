import kogitune
import os
import tempfile


def test_get_split():
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        kogitune.cli.get_split(
            dataset='kogi-jwu/jhumaneval',
            file_split=100,
        )

DUMMY="Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."

def make_dummy(filepath, N = 100000):
    import json
    dummy_text = '\n'.join([DUMMY] * ((N // len(DUMMY))+1))
    with open(filepath, "w") as w:
        for i in range(10000):
            d = {'text': dummy_text[:i]}
            print(json.dumps(d, ensure_ascii=False))

def test_store():
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        make_dummy('dummy.jsonl')
        kogitune.cli.store(
            files=['dummy.jsonl'], 
            tokenizer_path='llm-jp/llm-jp-3-1.8b',
            store_path='dummy',
            block_size=1024,
        )
