import json
from ..loads.commons import *

@adhoc.cli
def train_bpe_cli(**kwargs):
    from .tokenizers_bpe import train_bpe

    with adhoc.aargs_from(**kwargs) as aargs:
        files = aargs["files|!!"]
        save_path = aargs["save_path|!bpe"]
        train_bpe(files, save_path, **aargs)


@adhoc.from_kwargs
def word_list_from_kwargs(**kwargs):
    words = adhoc.get_list(kwargs, "word_list|words|!!")
    if len(words) == 1 and '.jsonl' in words[0]:
        path, args, tag = adhoc.parse_path(words[0], parent_args=kwargs)
        key = adhoc.get(args, 'key|=word')
        words = []
        with open(path) as f:
            for line in f.readlines():
                d = json.loads(line)
                try:
                    words.append(d[key])
                except KeyError as e:
                    adhoc.print('存在するkeyをちょうだい', d)
                    raise e
        return words
    if len(words) == 1 and words[0].endswith('.txt'):
        path, args, _ = adhoc.parse_path(words[0], parent_args=kwargs)
        words = []
        with open(path) as f:
            for line in f.readlines():
                if line.startswith('#'):
                    continue
                    words.append(line.strip())
        return words
    return words  


@adhoc.cli
def add_multi_token_cli(**kwargs):
    from .tokenizers_mte import make_mte

    with adhoc.aargs_from(**kwargs) as aargs:
        tokenizer = adhoc.load('from_kwargs', 'tokenizer', **aargs)
        words = adhoc.load('from_kwargs', 'word_list', **aargs)
        save_path = aargs["save_path|!mte"]
        bases = listfy(aargs['base|!<0x00>'])
        start=aargs['start|=10000']
        end=aargs['end']
        tokenizer = make_mte(tokenizer, words, bases, start=start, end=end)
        tokenizer.save_pretrained(save_path)

@adhoc.cli
def add_vocab_cli(**kwargs):
    add_multi_token_cli(**kwargs)


@adhoc.cli
def split_dataset_cli(**kwargs):
    """
    巨大なデータセットを複数ファイルに分割します。

    - dataset（必須）: データセットの指定
    - max_items=1000000: １ファイルあたりの最大データ件数
    - output_file: 出力先ファイル (拡張子は .jsonl.zst がおすすめ)
    - filter: 簡易的なフィルターも指定できます
    """
    import json
    from .file_spliters import FileSpliter
    
    with adhoc.aargs_from(**kwargs) as aargs:
        dataset = aargs["dataset|dataset_source"]
        start = aargs["start|=0"]
        end = aargs["end|head"]
        max_items = aargs["max_items|max_item|max|=1000000"]
        datastream = adhoc.load("datastream", dataset)
        config = {"data_source": dataset}

        # 簡易的なフィルター機能を持っている
        filter = adhoc.load('from_kwargs', 'filter', **aargs)
        if 'filter' in aargs:
            config["filter"] = filter.encode_as_json()

        if ".json" in dataset:
            output_file = basename(dataset, split_dir=False)
            output_file = aargs[f"output_file|={output_file}.jsonl.zst"]
        else:
            output_file = aargs[f"output_file|={basename(dataset)}.jsonl.zst"]

        with FileSpliter(output_file, config, max_items) as splitter:
            for sample in datastream.samples(start, end):
                sample = filter(sample)
                if sample is not None:
                    splitter.write(json.dumps(sample, ensure_ascii=False))


@adhoc.cli
def store_count_token_cli(**kwargs):
    from kogitune.trainers.recipe import DatasetRecipe
    import pandas as pd

    tqdm = adhoc.safe_import('tqdm')
    with adhoc.aargs_from(kwargs) as aargs:
        recipe = adhoc.load('from_kwargs', 'recipe', **aargs)
        tokenizer = adhoc.load('from_kwargs', 'tokenizer', **aargs)
        token_ids = list(range(0, tokenizer.vocab_size))
        vocabs = tokenizer.convert_ids_to_tokens(token_ids)
        counts = [0] * tokenizer.vocab_size
        ds = DatasetRecipe(recipe, **kwargs)
        for i in tqdm.auto.trange(len(ds)):
            block = ds[i] # numpy
            for token_id in block:
                counts[token_id] += 1
        # token_usage_statistics.csv
        output_file = aargs["output_file|output|!token_stat.csv"]
        df = pd.DataFrame({"token": vocabs, "count": counts})
        print(df["counts"].describe())
        df.to_csv(output_file)
