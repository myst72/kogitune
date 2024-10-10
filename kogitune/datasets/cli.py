import json
from ..loads.commons import *

@adhoc.cli
def train_bpe_cli(**kwargs):
    from .tokenizers_bpe import train_bpe

    with adhoc.kwargs_from_stacked(**kwargs) as kwargs:
        files = adhoc.get(kwargs, "files|!!")
        save_path = adhoc.get(kwargs, "save_path|!bpe")
        train_bpe(files, save_path, **kwargs)


@adhoc.cli
def add_vocab_cli(**kwargs):
    from .tokenizers_mte import make_mte

    with adhoc.kwargs_from_stacked(**kwargs) as kwargs:
        tokenizer = adhoc.load('from_kwargs', 'tokenizer', **kwargs)
        words = adhoc.load('from_kwargs', 'word_list', **kwargs)
        save_path = adhoc.get(kwargs, f"save_path|!{basename(tokenizer.name_or_path)}_mte")
        bases = adhoc.get_list(kwargs, 'multi_index|!<0x00>')
        start=adhoc.get(kwargs, 'start|=10000')
        end=adhoc.get(kwargs, 'end')
        tokenizer, table = make_mte(tokenizer, words, bases, start=start, end=end, **kwargs)
        tokenizer.save_pretrained(save_path)
        save_table('add_tokens.csv', table, save_path=save_path)
        
@adhoc.cli
def add_multi_token_cli(**kwargs):
    add_vocab_cli(**kwargs)




@adhoc.cli
def get_cli(**kwargs):
    """
    巨大なデータセットを複数ファイルに分割します。

    - dataset（必須）: データセットの指定
    - split=1000000: １ファイルあたりの最大データ件数
    - output_file: 出力先ファイル (拡張子は .jsonl.zst がおすすめ)
    - filter: 簡易的なフィルターも指定できます
    """
    import json
    from .file_spliters import FileSpliter
    
    with adhoc.kwargs_from_stacked(**kwargs) as kwargs:
        dataset = adhoc.get(kwargs, "dataset|dataset_source")
        start = adhoc.get(kwargs, "start|=0")
        end = adhoc.get(kwargs, "end|head")
        max_items = adhoc.get(kwargs, "split|!1000000")
        datastream = adhoc.load("datastream", dataset, **kwargs)
        config = {"data_source": dataset}

        # 簡易的なフィルター機能を持っている
        filter = adhoc.load('from_kwargs', 'filter', **kwargs)
        if 'filter' in kwargs:
            config["filter"] = filter.encode_as_json()

        if ".json" in dataset:
            output_file = basename(dataset, split_dir=False)
            output_file = adhoc.get(kwargs, f"output_file|={output_file}.jsonl.zst")
        else:
            output_file = adhoc.get(kwargs, f"output_file|={basename(dataset)}.jsonl.zst")

        with FileSpliter(output_file, config, max_items) as splitter:
            for sample in datastream.samples(start, end):
                sample = filter(sample)
                if sample is not None:
                    splitter.write(json.dumps(sample, ensure_ascii=False))


def store_pack_cli(**kwargs):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from ..datasets.chunks import store

    with adhoc.kwargs_from_stacked(**kwargs) as kwargs:
        base_dir = adhoc.get(kwargs, "save_path|!!")
        files = adhoc.get_list(kwargs, "files|!!")
        data_list = [{'_file': file, **kwargs} for file in files]
        num_workers = adhoc.get(kwargs, "num_workers|=1")
        if num_workers == 1:
            for data in data_list:
                store(data)
        else:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(store, data) for data in data_list]        
            for future in as_completed(futures):
                result = future.result()
                print(f"結果: {result}")

@adhoc.cli
def store_count_token_cli(**kwargs):
    from kogitune.trainers.recipe import DatasetRecipe
    import pandas as pd

    tqdm = adhoc.safe_import('tqdm')
    with adhoc.kwargs_from_stacked(kwargs) as kwargs:
        recipe = adhoc.load('from_kwargs', 'recipe', **kwargs)
        tokenizer = adhoc.load('from_kwargs', 'hftokenizer', **kwargs)
        token_ids = list(range(0, tokenizer.vocab_size))
        vocabs = tokenizer.convert_ids_to_tokens(token_ids)
        counts = [0] * tokenizer.vocab_size
        ds = DatasetRecipe(recipe, **kwargs)
        for i in tqdm.auto.trange(len(ds)):
            block = ds[i] # numpy
            for token_id in block:
                counts[token_id] += 1
        # token_usage_statistics.csv
        output_file = adhoc.get(kwargs, "output_file|output|!token_stat.csv")
        df = pd.DataFrame({"token": vocabs, "count": counts})
        print(df["counts"].describe())
        df.to_csv(output_file)

# def split_dataset_cli(**kwargs):
#     from kogitune.loads.datasets import split_dataset_cli

#     split_dataset_cli(**kwargs)



# # tokenizer


# ## filter 系

# def filter_cli(**kwargs):
#     from kogitune.filters.filters import filter_cli

#     filter_cli(**kwargs)


# def replace_cli(**kwargs):
#     from kogitune.filters.replaces import replace_cli

#     replace_cli(**kwargs)


# def filter_maxmin_cli(**kwargs):
#     from kogitune.filters.maxmins import filter_maxmin_cli

#     filter_maxmin_cli(**kwargs)


# def filter_langset_cli(**kwargs):
#     from kogitune.filters.languages import filter_langset_cli

#     filter_langset_cli(**kwargs)


# ## store 系




# def store_cli(**kwargs):
#     from .stores import store_files

#     with adhoc.kwargs_from_stacked(**kwargs) as kwargs:
#         files = adhoc.get(kwargs, "files|!!ファイルを一つ以上与えてください"]
#         store_files(files, skip_validation=False)


# def head_cli(**kwargs):
#     from .trainers import DatasetComposer

#     with DatasetComposer(**kwargs) as dc:
#         dc.with_format("numpy")
#         start = dc.adhoc.get(kwargs, "start|=0"]
#         N = dc.adhoc.get(kwargs, "head|N|batch|=1024"]
#         tokenizer = dc.get_tokenizer()
#         ds = dc.get_train_dataset()
#         for i in range(start, start + N):
#             example = ds[i]
#             if "input_ids" in example:
#                 print(f"inputs[{i}]:", tokenizer.decode(example["input_ids"]))
#             if "labels" in example:
#                 print(f"labels[{i}]:", tokenizer.decode(example["labels"]))
#             print("---")


# FREEZE = """
# from datasets import load_from_disk
# ds = load_from_disk("{}")
# """


# def freeze_cli(**kwargs):
#     import time
#     from datasets import Dataset
#     from .trainers import DatasetComposer

#     input_ids = []
#     attention_mask = []
#     labels = []
#     start = time.time()
#     with DatasetComposer(prefetch=0, **kwargs) as dc:
#         dc.with_format("tensor")
#         ds = dc.get_train_dataset()
#         for i in adhoc.tqdm(range(len(ds))):
#             example = ds[i]
#             input_ids.append(example["input_ids"])
#             if "attention_mask" in example:
#                 attention_mask.append(example["attention_mask"])
#             if "labels" in example:
#                 labels.append(example["labels"])
#             if len(labels) > 0:
#                 ds_dict = {
#                     "input_ids": input_ids,
#                     "attention_mask": attention_mask,
#                     "labels": labels,
#                 }
#             elif len(attention_mask) > 0:
#                 ds_dict = {"input_ids": input_ids, "attention_mask": attention_mask}
#             else:
#                 ds_dict = {"input_ids": input_ids}
#         adhoc.print(f"ダウンロード時間: {time.time()-start} s")
#         ds = Dataset.from_dict(ds_dict).with_format("torch")
#         print(ds)
#         output_path = dc.adhoc.get(kwargs, "output_path|!freezed_dataset"]
#         ds.save_to_disk(output_path)
#         print(FREEZE.format(output_path))


# def token_stat_cli(**kwargs):
#     import pandas as pd
#     from .trainers import DatasetRecipe

#     with DatasetRecipe(prefetch=0, **kwargs) as dc:
#         dc.with_format("numpy")
#         tokenizer = dc.get_tokenizer()
#         token_ids = list(range(0, tokenizer.vocab_size))
#         vocabs = tokenizer.convert_ids_to_tokens(token_ids)
#         counts = [0] * tokenizer.vocab_size
#         ds = dc.get_train_dataset()
#         for i in adhoc.tqdm(range(len(ds)), desc="counting tokens"):
#             example = ds[i]
#             for token_id in example["input_ids"]:
#                 counts[token_id] += 1
#             if "labels" in example:
#                 for token_id in example["labels"]:
#                     counts[token_id] += 1
#         output_file = dc.adhoc.get(kwargs, "output_file|output|=token_stat.csv"]
#         df = pd.DataFrame({"token": vocabs, "count": counts})
#         print(df["counts"].describe())
#         df.to_csv(output_file)
#         adhoc.print(
#             f"トークンの出現回数を output_file='{output_file}' に保存しました。ふむふむ"
#         )


# ## 事前学習系


# def scratch_cli(**kwargs):
#     from kogitune.trainers.scratch import scratch_cli

#     scratch_cli(**kwargs)


# def pretrain_cli(**kwargs):
#     import torch

#     torch.backends.cuda.matmul.allow_tf32 = True
#     from kogitune.trainers import DatasetComposer

#     with DatasetComposer(**kwargs) as dc:
#         dc.train()


# def data_cli(**kwargs):
#     from kogitune.metrics import load_data

#     with adhoc.kwargs_from_stacked(**kwargs) as kwargs:
#         datalist = load_data(kwargs)


# def test_model_cli(**kwargs):
#     from kogitune.metrics.OLDmodels import test_model_cli

#     test_model_cli(**kwargs)


# def generate_cli(**kwargs):
#     from kogitune.metrics.OLDmodels import test_model_cli

#     test_model_cli(**kwargs)


# def convert_dataset_cli(**kwargs):
#     from kogitune.datasets.cli import convert_dataset_cli

#     convert_dataset_cli(**kwargs)


# def finetune_cli(**kwargs):
#     from kogitune.trainers import finetune_cli

#     finetune_cli(**kwargs)


# ## 評価系


# def chaineval_cli(**kwargs):
#     from kogitune.metrics.chaineval import chaineval_cli

#     chaineval_cli(**kwargs)


# def eval_cli(**kwargs):
#     from kogitune.metrics.chaineval import chaineval_cli

#     chaineval_cli(**kwargs)


# def eval_loss_cli(**kwargs):
#     from kogitune.metrics.chaineval import eval_loss_cli

#     eval_loss_cli(**kwargs)


# def eval_choice_cli(**kwargs):
#     from kogitune.metrics.chaineval import eval_choice_cli

#     eval_choice_cli(**kwargs)


# def self_check_cli(**kwargs):
#     from kogitune.metrics.chaineval import selfcheck_cli

#     selfcheck_cli(**kwargs)


# def selfcheck_cli(**kwargs):
#     from kogitune.metrics.chaineval import selfcheck_cli

#     selfcheck_cli(**kwargs)


# def import_output_cli(**kwargs):
#     "実験結果をインポートする"
#     from kogitune.metrics.OLDsamples import import_output_cli

#     import_output_cli(**kwargs)


# def jsonlfy_dataset_cli(**kwargs):
#     "実験結果をインポートする"
#     from kogitune.metrics.OLDsamples import jsonlfy_dataset_cli

#     jsonlfy_dataset_cli(**kwargs)
