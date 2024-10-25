from typing import List
import itertools

from .commons import *
from .files import file_jsonl_reader
from .tables import parse_pandas_extention, read_samples_from_pandas

class DataStreamLoader(adhoc.AdhocLoader):

    def load_from_map(self, path, kwargs:dict):
        if ".json" in path:
            return JSONLDataStream(**kwargs)
        if parse_pandas_extention(path) != '':
            return PandasStream(**kwargs)
        name = kwargs.pop('name', None)
        name = kwargs.pop('_name', None) or name
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
        self.get(kwargs, "dataset_start|=0", "dataset_end|head")
        self.init_kwargs = {**kwargs}

    @property
    def datatag(self):
        return basename(self.path)

    def samples(self, start=0, end=None):
        start = start if start != 0 else self.dataset_start
        end = end or self.dataset_end
        return self.read_stream(start, end)

class JSONLDataStream(DataStream):

    def read_stream(self, start, end):
        return file_jsonl_reader(self.path, start, end)

class PandasStream(DataStream):

    def read_stream(self, start, end):
        args = {**self.init_kwargs}
        if start > 0:
            args['skiprows'] = start
        if end:
            args['nrows'] = (end - start)
        args['chunksize'] = 4096
        samples = read_samples_from_pandas(self.path, **args)
        return iter(samples)


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

def _load_hfdataset(path, args, imported_datasets):
    adhoc.verbose_print('Loading//ロード中', path, args, once=path)
    adhoc.verbose_print('強制的にオプションを追加するには、DATASET_xxx=yyy', once='DATASET', face='  ')

    try:
        dataset = imported_datasets.load_dataset(path, **args)
    except BaseException as e:
        adhoc.report_ArgumentError(message='Failed to loading///データセットのロード失敗', 
                                       throw=e, 
                                       called=adhoc.function_called(
                                           'datasets.load_dataset', path, args))
    return _select_dataset_split(dataset, imported_datasets)
        
def _select_dataset_split(dataset, imported_datasets):
    if isinstance(dataset, imported_datasets.DatasetDict):
        keys = list(dataset.keys())
        adhoc.verbose_print(f'splitの指定がないから、split="{keys[0]}"を使うよ', once=f'split={keys[0]}')
        dataset = iter(dataset[keys[0]])
    if isinstance(dataset, imported_datasets.IterableDatasetDict):
        keys = list(dataset.keys())
        adhoc.verbose_print(f'splitの指定がないから、split="{keys[0]}"を使うよ', once=f'split={keys[0]}')
        dataset = dataset[keys[0]]
    return dataset

class HFDatasetStream(DataStream):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.suffix = ''
        self.pathargs = adhoc.safe_kwargs(kwargs, load_dataset_kw, unsafe='DATASET')

    @property
    def datatag(self):
        datatag = self.tag if self.tag != '' else basename(self.path, split_ext=False)
        if datatag.startswith("openai_"):
            datatag = datatag[len("openai_") :]
        return f"{datatag}{self.suffix}"

    def stream(self):
        datasets = adhoc.safe_import('datasets')

        if 'name' in self.pathargs:
            name = self.pathargs['name']
            self.suffix = f'_{name}'

        dataset = _load_hfdataset(self.path, self.pathargs, datasets)
        for item in dataset:
            yield {**item}

    def read_stream(self, start, end):
        if start != 0 or end != None:
            return itertools.islice(self.stream(), start, end)
        return self.stream()

class HFDatasetNames(HFDatasetStream):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.subset_names = adhoc.get_list(kwargs, "dataset_subset")

    def stream(self):
        datasets = adhoc.safe_import('datasets')

        if self.subset_names:
            names = self.subset_names
        else:
            names = load_dataset_names(self.path, datasets)

        for name in names:
            args = self.pathargs | {"name": name}
            dataset = _load_hfdataset(self.path, args, datasets)
            for item in dataset:
                yield {**item, "_group": name}

def load_dataset_names(path, imported_datasets=None):
    if 'JMMLU' in path:
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
    if imported_datasets:
        try:
            builder = imported_datasets.load_dataset_builder(path)
            # データセットのサブセット（バリエーション）の名前を取得
            if hasattr(builder.info, "config_names"):
                return builder.info.config_names
        except BaseException as e:
            adhoc.print('Please set `dataset_subset`//`dataset_subset`を指定してね')
            adhoc.exit(throw=e)
    return []


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
                format = format.replace(r"\n", "\n")
                self.rules.append((key, format))
        if columns is not None:
            self.columns = adhoc.list_keys(columns, sep=",")

    def isNullObject(self):
        return len(self.rules) == 0 and self.columns is None

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
        if isinstance(samples, list):
            for sample in samples:
                self.transform_s(sample)
            return samples
        if isinstance(samples, dict):
            return self.transform_s(samples)
        return TransformIter(self, samples)

@adhoc.from_kwargs
def transform_from_kwargs(**kwargs):
    transform = adhoc.get(kwargs, "dataset_transform|transform")
    columns = adhoc.get(kwargs,"dataset_columns|columns")
    return Transform(transform, columns)


class TransformIter(object):
    def __init__(self, transform, iterator):
        self.transform = transform
        self.iterator = iterator

    def __next__(self):
        return self.transform.transform_s(next(self.iterator))

def apply_template(template:dict, sample: dict, 
                   keys=List[str],
                   record: dict=None):
    record = sample if record is None else record
    for key, format in template.items():
        record[key] = format.format(**sample)
        
