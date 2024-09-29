from typing import List, Union
import itertools
import json
import os

from .commons import *
from .files import file_jsonl_reader, write_config, safe_makedirs


class DataStream(adhoc.LoaderObject):
    def __init__(self, path, kwargs):
        self.path = path
        self.pathargs = {}
        self.get(kwargs, "start|=0", "end|head")

    def datatag(self):
        return basename(self.path)

    def samples(self, start=0, end=None):
        start = self.start or start
        end = self.end or end
        # print("@stream", start, end)
        return self.stream(start, end)


class JSONLDataStream(DataStream):

    def stream(self, start=0, end=None):
        start = self.start or start
        end = self.end or end
        # print("@stream", start, end)
        return file_jsonl_reader(self.path, start, end)

class CSVDataStream(DataStream):

    def stream(self, start=0, end=None):
        import pandas as pd
        start = self.start or start
        end = self.end or end
        df = pd.read_csv(self.path)
        df.columns = [str(name).lower().replace(" ", "_") for name in df.columns]
        samplelist = df.to_dict(orient="records")
        return iter(samplelist[start:end])


load_dataset_kw = [
    "name",  # Optional[str] = None,
    "data_dir",  #: Optional[str] = None,
    "data_files",  #: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]] = None,
    "split",  #: Optional[Union[str, Split]] = None,
    "cache_dir",  #: Optional[str] = None,
    "keep_in_memory",  #: Optional[bool] = None,
    "save_infos",  #: bool = False,
    "revision",  #: Optional[Union[str, Version]] = None,
    "token",  #: Optional[Union[bool, str]] = None,
    # use_auth_token="deprecated",
    # task="deprecated",
    "streaming",  #: bool = False,
    "num_proc",  #: Optional[int] = None,
    # storage_options: Optional[Dict] = None,
]


class HFDatasetStream(DataStream):
    def __init__(self, path, kwargs):
        super().__init__(path, kwargs)
        self.suffix = ''
        self.pathargs = adhoc.safe_kwargs(kwargs, *load_dataset_kw)

    def datatag(self):
        datatag = basename(self.path, split_ext=False)
        if datatag.startswith("openai_"):
            datatag = datatag[len("openai_") :]
        return f"{datatag}{self.suffix}"

    def stream(self, start=0, end=None):
        datasets = adhoc.safe_import('datasets')

        if 'name' in self.pathargs:
            name = self.pathargs['name']
            self.suffix = f'_{name}'

        if 'split' not in self.pathargs:
            split_names = datasets.get_dataset_split_names(self.path, **self.pathargs)
            self.pathargs['split'] = split_names[0]
            adhoc.verbose_print(f'splitの指定がないから、split="{split_names[0]}"を使うよ', once=self.path)

        dataset = datasets.load_dataset(self.path, **self.pathargs)
        if isinstance(dataset, datasets.DatasetDict):
            keys = list(dataset.keys())
            dataset = iter(dataset[keys[0]])
        if isinstance(dataset, datasets.IterableDatasetDict):
            keys = list(dataset.keys())
            dataset = dataset[keys[0]]
        if start != 0 or end != None:
            return itertools.islice(dataset, start, end)
        return dataset


class HFDatasetNames(HFDatasetStream):

    def stream_names(self):
        import datasets

        names = adhoc.load("dataset_names", self.path, use_default=[])
        if len(names) == 0:
            # データセットのビルダーオブジェクトを取得
            builder = datasets.load_dataset_builder(self.path)
            # データセットのサブセット（バリエーション）の名前を取得
            if hasattr(builder.info, "config_names"):
                names = builder.info.config_names

        if 'split' not in self.pathargs:
            split_names = datasets.get_dataset_split_names(self.path, **self.pathargs)
            self.pathargs['split'] = split_names[0]
            adhoc.verbose_print(f'splitの指定がないから、split="{split_names[0]}"を使うよ')

        for name in names:
            dataset = datasets.load_dataset(self.path, name, self.pathargs)
            if isinstance(dataset, datasets.dataset_dict.DatasetDict):
                split_list = list(dataset.keys())
                split = "test" if "test" in dataset else split_list[0]
                dataset = dataset[split]
            for item in dataset:
                yield {"group": name} | {k: v for k, v in item.items()}


    def stream(self, start=0, end=None):
        if start != 0 or end != None:
            return itertools.islice(self.stream_names(), start, end)
        return self.stream_names()

##
# loader



class DataStreamLoader(adhoc.AdhocLoader):
    def load(self, path: str, tag: str, kwargs):
        if ".json" in path:
            return JSONLDataStream(path, kwargs)
        if path.endswith(".csv"):
            return CSVDataStream(path, kwargs)
        
        name = kwargs.pop('name', None)
        if name is not None:
            if name == '*':
                return HFDatasetNames(path, kwargs)
            if name != '':
                kwargs['name'] = name
        return HFDatasetStream(path, kwargs)

DataStreamLoader().register("datastream")

class Transform(object):
    def __init__(self, transforms=None, columns=None):
        """
        transforms='new_key=old_key|remove_key'
        """
        self.rules = []
        self.columns = None
        if transforms is not None:
            for key in adhoc.list_keys(transforms):
                key, _, format = key.partition("=")
                format = transforms.replace(r"\n", "\n")
                self.rules.append((key, format))
        if columns is not None:
            self.columns = adhoc.list_keys(columns, sep=",")

    def transform_s(self, sample: dict):
        for key, format in self.rules:
            if format == "":
                del sample[key]
            elif "{" in format:
                sample[key] = format.format(**(sample))
            else:
                sample[key] = sample[format]
        if self.columns:
            source = sample.copy()
            for key in source.keys():
                sample.pop(key)
            for key in self.columns:
                sample[key] = source[key]
        return sample

    def transform(self, samples):
        if isinstance(samples, dict):
            return self.transform_s(samples)
        if isinstance(samples, list):
            for sample in samples:
                self.transform_s(sample)
            return samples
        return TransformIter(self, samples)

@adhoc.from_kwargs
def transform_from_kwargs(**kwargs):
    transform = adhoc.get(kwargs,"transform")
    columns = adhoc.get(kwargs,"columns*")
    return Transform(transform, columns)


class TransformIter(object):
    def __init__(self, transform, iterator):
        self.transform = transform
        self.iterator = iterator

    def __next__(self):
        return self.transform.transform_s(next(self.iterator))


class RecordDataLoader(adhoc.AdhocLoader):

    def load(self, path: str, tag: str, kwargs):
        import pandas as pd
        if path.endswith('.txt'):
            df = pd.read_table(path, sep='\n', header=None)
            df.columns = ['text']
            dict_rows = df.to_dict(orient='records')
            return RecordData(path, dict_rows)
        if path.endswith('.csv'):
            df = pd.read_csv(path)
            df.columns = [str(name).lower().replace(" ", "_") for name in df.columns]
            dict_rows = df.to_dict(orient='records')
            return RecordData(path, dict_rows)
        if path.endswith('.tsv'):
            df = pd.read_csv(path, sep='\t')
            df.columns = [str(name).lower().replace(" ", "_") for name in df.columns]
            dict_rows = df.to_dict(orient='records')
            return RecordData(path, dict_rows)
        if path.endswith('.xlsx'):
            df = pd.read_excel(path)
            df.columns = [str(name).lower().replace(" ", "_") for name in df.columns]
            dict_rows = df.to_dict(orient='records')
            return RecordData(path, dict_rows)
        if path.endswith('.xls'):
            df = pd.read_excel(path)
            df.columns = [str(name).lower().replace(" ", "_") for name in df.columns]
            dict_rows = df.to_dict(orient='records')
            return RecordData(path, dict_rows)
        df = pd.read_json(path, lines=True)
        dict_rows = df.to_dict(orient='records')
        return RecordData(path, dict_rows)


RecordDataLoader().register("record")

def rename_path_as_jsonl(path):
    if not os.path.exists(path):
        return f"{basename(path)}.jsonl"
    else:
        return f"{basename(path, split_dir=False)}.jsonl"

class RecordData(adhoc.LoaderObject):
    def __init__(self, path: str, samplelist: List[dict]):
        self.path = path
        self.samplelist = samplelist
        self.save_path = rename_path_as_jsonl(path)
        self.save_mode = "w"

    def samples(self, start=0, end=None):
        return self.samplelist[start:end]

    def get_sample(self, *keys):
        sample = self.samplelist[0]
        values = [sample.get(key, "") for key in keys]
        return values[0] if len(keys) == 1 else values

    def save(self, save_path=None):
        save_path = save_path or self.save_path

        if save_path:
            safe_makedirs(save_path)
            with open(save_path, mode=self.save_mode, encoding="utf-8") as w:
                for result in self.samplelist:
                    assert isinstance(result, dict)
                    print(json.dumps(result, ensure_ascii=False), file=w)

    def rename_save_path(self, **kwargs):
        head = adhoc.get(kwargs, "test_run|head")
        if head:
            self.save_path = None
            return head
        output_path = adhoc.get(kwargs, "output_path")
        if output_path:
            self.save_path = os.path.join(output_path, basename(self.save_path, split_ext=False))
            return None
        output_file = adhoc.get(kwargs, "output_file")
        if output_file:
            if os.path.exists(output_file):
                os.unlink(output_file)
            self.save_path = output_file
            self.save_mode="a"
            return None
        overwrite = adhoc.get(kwargs, "overwrite|=False")
        if overwrite == False and output_file is None:
            adhoc.print(
                "To save a file, you need `output_file` or `overwrite=True`"
                "//ファイル出力するには、output_fileかoverwrite=Trueを指定しよう"
            )
            self.save_path = None
            return 5
        return None



