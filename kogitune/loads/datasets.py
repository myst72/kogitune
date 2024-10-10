from typing import List
import itertools
import json
import os
import pandas as pd

from .commons import *
from .files import file_jsonl_reader, safe_makedirs



class DataStreamLoader(adhoc.AdhocLoader):

    def load_from_map(self, path, kwargs:dict):
        if ".json" in path:
            return JSONLDataStream(**kwargs)
        if path.endswith(".csv"):
            return CSVDataStream(**kwargs)
        name = kwargs.pop('name', None)
        if name is not None:
            if name == '*':
                return HFDatasetNames(**kwargs)
            if name != '':
                kwargs['name'] = name
        return HFDatasetStream(**kwargs)

DataStreamLoader({}).register("datastream|dataset")


class DataStream(adhoc.AdhocObject):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.pathargs = {}
        self.get(kwargs, "start|=0", "end|head")

    @property
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
    "trust_remote_code", 
    # storage_options: Optional[Dict] = None,
]

def _load_hfdataset(path, args, datasets):
    adhoc.print('Loading///ロード中', path, args, once=path)
    try:
        dataset = datasets.load_dataset(path, **args)
    except BaseException as e:
        adhoc.report_ArgumentError(message='Failed to loading///データセットのロード失敗', 
                                       throw=e, 
                                       called=adhoc.function_called(
                                           'datasets.load_dataset', path, args))
    return select_dataset_split(dataset, datasets)
        
def select_dataset_split(dataset, datasets):
    if isinstance(dataset, datasets.DatasetDict):
        keys = list(dataset.keys())
        adhoc.verbose_print(f'splitの指定がないから、split="{keys[0]}"を使うよ', once=f'{dataset}')
        dataset = iter(dataset[keys[0]])
    if isinstance(dataset, datasets.IterableDatasetDict):
        keys = list(dataset.keys())
        adhoc.verbose_print(f'splitの指定がないから、split="{keys[0]}"を使うよ', once=f'{dataset}')
        dataset = dataset[keys[0]]
    return dataset


class HFDatasetStream(DataStream):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.suffix = ''
        self.pathargs = adhoc.safe_kwargs(kwargs, load_dataset_kw)

    @property
    def datatag(self):
        datatag = self.tag if self.tag != '' else basename(self.path, split_ext=False)
        if datatag.startswith("openai_"):
            datatag = datatag[len("openai_") :]
        return f"{datatag}{self.suffix}"

    def stream_tagging(self, start=0, end=None):
        datasets = adhoc.safe_import('datasets')

        if 'name' in self.pathargs:
            name = self.pathargs['name']
            self.suffix = f'_{name}'

        dataset = _load_hfdataset(self.path, self.pathargs, datasets)
        for item in dataset:
            yield {"dataset": self.datatag, **item}

    def stream(self, start=0, end=None):
        if start != 0 or end != None:
            return itertools.islice(self.stream_tagging(), start, end)
        return self.stream_tagging()


class HFDatasetNames(HFDatasetStream):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.subset_names = adhoc.get_list(kwargs, "dataset_subset")

    def stream_tagging(self):
        datasets = adhoc.safe_import('datasets')

        if self.subset_names:
            names = self.subset_names
        else:
            names = self.load_dataset_names(datasets)

        for name in names:
            args = self.pathargs | {"name": name}
            dataset = _load_hfdataset(self.path, args, datasets)
            for item in dataset:
                yield {"dataset": self.datatag, "group": name, **item} #| {k: v for k, v in item.items()}

    def load_dataset_names(self, datasets):
        if 'JMMLU' in self.path:
            return ['japanese_history', 'miscellaneous', 'security_studies', 'virology', 'nutrition', 
                    'human_sexuality', 'college_mathematics', 'japanese_civics', 'econometrics', 
                    'computer_security', 'clinical_knowledge', 'machine_learning', 'high_school_chemistry',
                    'human_aging', 'logical_fallacies', 'sociology', 'high_school_european_history', 
                    'high_school_statistics', 'high_school_physics', 'high_school_microeconomics', 
                    'college_physics', 'anatomy', 'high_school_psychology', 'business_ethics', 
                    'professional_psychology', 'college_medicine', 'elementary_mathematics', 
                    'moral_disputes', 'marketing', 'high_school_macroeconomics', 'world_religions', 
                    'conceptual_physics', 'professional_medicine', 'prehistory', 'high_school_mathematics', 
                    'international_law', 'philosophy', 'japanese_idiom', 'japanese_geography', 'management',
                      'high_school_computer_science', 'medical_genetics', 'college_computer_science', 
                      'public_relations', 'professional_accounting', 'abstract_algebra', 'global_facts', 
                      'college_biology', 'high_school_geography', 'world_history', 'high_school_biology', 
                      'college_chemistry', 'electrical_engineering', 'astronomy', 'jurisprudence', 'formal_logic']

        try:
            builder = datasets.load_dataset_builder(self.path)
            # データセットのサブセット（バリエーション）の名前を取得
            if hasattr(builder.info, "config_names"):
                return builder.info.config_names
        except BaseException as e:
            adhoc.print('Please set `dataset_subset`///`dataset_subset`を指定してね')
            adhoc.exit(throw=e)



@adhoc.from_kwargs
def iterate_datasets_from_kwargs(**kwargs):
    scheme = kwargs.pop('_scheme', 'testdata')
    dataset_list = adhoc.get_list(kwargs, "dataset_list|dataset|!!")
    for path in dataset_list:
        if 'dataset_subset' in kwargs:
            path, largs, _ = adhoc.parse_path(path, parent_args=kwargs)
            if largs.get('name') != '*':
                for subset_name in adhoc.get_list(kwargs, "dataset_subset|="):
                    dataset = adhoc.load(scheme, path, **(kwargs|{'_name': subset_name}))
                    yield dataset
                continue
        dataset = adhoc.load(scheme, path, **kwargs)
        yield dataset


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


