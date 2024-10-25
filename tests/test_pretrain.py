import kogitune
import os
import tempfile

DUMMY="Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."

def make_dummy(filepath, N = 100000):
    import json
    dummy_text = '\n'.join([DUMMY] * ((N // len(DUMMY))+1))
    with open(filepath, "w") as w:
        for i in range(10000):
            d = {'text': dummy_text[:i]}
            print(json.dumps(d, ensure_ascii=False), file=w)

def test_pretrain():
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        make_dummy('dummy.jsonl', N=1000)
        kogitune.cli.store(
            files=['dummy.jsonl'], 
            tokenizer_path='llm-jp/llm-jp-3-1.8b',
            store_path='dummy',
            block_size=1024,
        )
        kogitune.cli.pretrain(
            recipe=['dummy'],
            tokenizer_path='llm-jp/llm-jp-3-1.8b',
            batch_size=16,
            block_size=32,
            model_path='kkuramitsu/chico-0.03b',
            save_path='pretrain',
            max_steps=2,
            use_wandb=False,
        )
        assert os.path.isdir('pretrain')

def test_pretrain_recipe():
    from kogitune.trainers.recipe import DatasetRecipe
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        url_list = [
            "https://papertown-jwu.s3.ap-northeast-1.amazonaws.com/llm-jp-915a/mc4ja_line?length=1b",
            "https://papertown-jwu.s3.ap-northeast-1.amazonaws.com/llm-jp-915a/pythondoc_def",
            "https://papertown-jwu.s3.ap-northeast-1.amazonaws.com/llm-jp-915a/python?ratio=0.1",
            "https://papertown-jwu.s3.ap-northeast-1.amazonaws.com/llm-jp-915a/markdownpyedu",
        ]
        dataset = DatasetRecipe(url_list, batch_size=1024, block_size=512)
        for i in range(100):
            print(dataset[i])

