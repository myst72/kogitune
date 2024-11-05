from typing import List
import os
import json
import math
import numpy as np
import pandas as pd
from ..loads.commons import *

## Record

class RecordLoader(adhoc.AdhocLoader):

    def load_from_map(self, path, kwargs):
        dict_rows = read_samples_from_pandas(path, **kwargs)
        return RecordData(path, dict_rows, **kwargs)

RecordLoader({}).register("record")

PANDAS_CSV_KWARGS=[
    'sep', #**`sep`**：列を区切る区切り文字を指定します。
    'header', # 列名として使用する行を指定します。`header=None` とすると
    'names', # 列名を手動で指定します。`header=None`と組み合わせて使用する
    'usecols', # 特定の列だけを読み込みたい場合に指定します。
    'na_values', # 特定の値を `NaN`（欠損値）として扱うように指定します。
    'skiprows', # 最初の数行をスキップして読み込む場合に使用します。
    'nrows', #読み込む行数を指定します。
    'encoding', # ファイルのエンコーディングを指定します。
    'sheetname', # エクセル用
]

def read_samples_from_pandas(path, /, **kwargs):
    verbose = len(kwargs) > 0
    ext = parse_pandas_extention(path)
    if path in ('txt'):
        names = kwargs.pop('names', ['text'])
        kwargs = adhoc.safe_kwargs(kwargs, PANDAS_CSV_KWARGS, unsafe='PANDAS')
        kwargs = kwargs | { 'header': None }
        df = pd.read_table(path, **kwargs)
        dict_rows = []
        for i in range(len(df)):
            word = df.loc[i][0]
            if not word.startswith('#'):
                dict_rows.append({names[0]: word})
        return dict_rows
    if ext == 'jsonl':
        df = pd.read_json(path, lines=True)
        return df.to_dict(orient='records')
    if ext in ('csv', 'tsv') or path.endswith('?format=csv'):
        # Googleスプレッドシートの共有リンク
        # 共有リンクの「/edit」部分を「/export?format=csv」に変更
        # sheet_url = "https://docs.google.com/spreadsheets/d/your_spreadsheet_id/export?format=csv"
        kwargs = adhoc.safe_kwargs(kwargs, PANDAS_CSV_KWARGS, unsafe='PANDAS')
        if ext == 'tsv':
            kwargs['sep'] = '\t'
        kwargs.pop('sheetname', None)
        df = pd.read_csv(path, **kwargs)
        if 'names' not in kwargs:
            df.columns = [str(name).lower().replace(" ", "_") for name in df.columns]
        if verbose:
            adhoc.verbose_print(path)
            adhoc.verbose_print(df.head(2), face='')
        return df.to_dict(orient='records')
    if ext in ('xls', 'xlsx'):
        kwargs = adhoc.safe_kwargs(kwargs, PANDAS_CSV_KWARGS, unsafe='PANDAS')
        df = pd.read_excel(path, **kwargs)
        if 'names' not in kwargs:
            df.columns = [str(name).lower().replace(" ", "_") for name in df.columns]
        if verbose:
            adhoc.verbose_print(path)
            adhoc.verbose_print(df.head(2), face='')
        return df.to_dict(orient='records')
    adhoc.exit(throw=ValueError(f'Unsupported file type: {path}'))

def parse_pandas_extention(path, skip_z=True):
    if '.' in path:
        path, _, ext = path.rpartition('.')
        if skip_z and ext in ('zip', 'gz', 'bz2', 'xz', 'zst'):
            return parse_pandas_extention(path, skip_z=False)
        return ext
    return ''


def rename_path_as_jsonl(path, **kwargs):
    if kwargs.get('_readonly', False):
        return None
    output_path = adhoc.get(kwargs, "output_path|output_dir")
    if output_path:
        path = os.path.join(output_path, f"{basename(path)}.jsonl")
    else:
        path = f"{basename(path, split_dir=False)}.jsonl"
    return path


class RecordData(adhoc.AdhocObject):
    def __init__(self, path: str, samplelist: List[dict], **kwargs):
        self.path = path
        self.save_path = rename_path_as_jsonl(path, **kwargs)
        resume = adhoc.get(kwargs, 'resume|_resume|=True')
        if self.save_path and os.path.exists(self.save_path) and resume :
            df = pd.read_json(self.save_path, lines=True)
            self.samplelist = df.to_dict(orient='records')
            adhoc.verbose_print(f'既存のファイル{self.save_path}を使います。嫌なら`resume=False`してね', once=f"`resume=False`")            
        else:
            self.samplelist = samplelist
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
        output_path = adhoc.get(kwargs, "output_path|output_dir")
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

    @classmethod
    def filter_samples(cls, samples: List[dict], conditions: List[str]):
        matched_samples = []
        for sample in samples:
            matched=True
            for condition in conditions:
                if not safe_sample_matched(sample, condition):
                    matched = False
                    break
            if matched:
                matched_samples.append(sample)   
        return matched_samples

    @classmethod
    def extract_labels(cls, samples: List[dict], condition:str):
        labels = []
        for sample in samples:
            labels.append(int(safe_sample_matched(sample, condition)))
        return labels

    @classmethod
    def extract_values(cls, samples: List[dict], key:str):
        return [sample[key] for sample in samples]

    @classmethod
    def update_kwargs(self, samples:List[dict], /, **kwargs):
        items = list(kwargs.items())
        for sample in samples:
            for k, v in items:
                sample[k] = v

    @classmethod
    def update_values(self, samples: List[dict], results: dict):
        """
        results = {"output": scores}
        """
        for key, outputs in results.items():
            if isinstance(outputs, tuple):
                # ('mean', scores) 形式への対応
                outputs = outputs[1]
            assert len(outputs) == len(samples)
            for i, sample in enumerate(samples):
                sample[key] = outputs[i]

def safe_sample_matched(sample:dict, expr):
    #print('@expr', expr, sample)
    return eval(expr, None, {'sample': sample})


@adhoc.from_kwargs
def word_list_from_kwargs(**kwargs):
    words = adhoc.get_list(kwargs, "word_list|words|!!")
    if len(words) == 1:
        is_text_file = '.txt' in words[0]
        path, args, tag = adhoc.parse_path(words[0], parent_args=kwargs)
        args['_readonly'] = True
        record = adhoc.load('record', path, **args)
        key = adhoc.get(args, 'word_key|key|=word')
        words = []
        for sample in record.samples():
            text = adhoc.get_formatted_text(sample, key)
            if is_text_file and text.startswith('#'):
                continue
            words.append(text)
        return words
    return words


## LeaderBoard

class LeaderBoard(object):
    def __init__(self, filepath="leaderboard.csv", /, **kwargs):
        self.tablepath = filepath
        self.scorepath = filepath.replace('.csv', '_score.jsonl')
        safe_makedirs(self.tablepath)
        safe_makedirs(self.scorepath)
        adhoc.get(kwargs, "overwrite")

    def update(self, index: str, key: str, value: float):
        found = False
        if os.path.exists(self.tablepath):
            df = pd.read_csv(self.tablepath)
            json_list = df.to_dict(orient="records")
            for d in json_list:
                if d["index"] == index:
                    d[key] = value
                    found = True
        else:
            json_list = []
        if found == False:
            json_list.append(
                {
                    "index": index,
                    key: value,
                }
            )
        df = pd.DataFrame(json_list)
        df.to_csv(self.tablepath, index=False)

    def append_score(self, record: dict):
        with open(self.scorepath, "a", encoding="utf-8") as w:
            print(json.dumps(record, ensure_ascii=False), file=w)

    def pivot_table(self, samples:dict, name:str, aggfunc = 'mean', /, **kwargs):
        if isinstance(name, dict):
            results = name
            result = None
            for name in results.keys():
                aggfunc = None
                if isinstance(results[name], str):
                    aggfunc = results[name]
                elif isinstance(results[name], tuple):
                    aggfunc = results[name][0]  # ('mean', scores) 形式
                if aggfunc:
                    result = self.pivot_table(samples, name, aggfunc, **kwargs)
            return result
        
        groupby = kwargs.get('groupby', kwargs.get('groupby', None))
        grouped_scores = self.get_grouped_scores(samples, name, groupby)

        label_query = None
        if aggfunc.startswith('AUC:') or aggfunc.startswith('ROC:'):
            label_query = aggfunc[4:]
            if "sample[" not in label_query:
                label_query = f'sample["{label_query}"]'

        for key, scores in grouped_scores.items():
            model, datatag, group = key
            value_name = self.get_value_name(datatag, group, name)
            record = {
                "model": model,
                "dataset": datatag,
                "group": group,
                "metric": name,
                "value_name": value_name,
                "count": len(scores),
            }
            if label_query:
                conditions = [
                    f'sample["_model"] == "{model}"',
                    f'sample["_dataset"] == "{datatag}"',
                ]
                if groupby:
                    conditions.append(f'sample["{groupby}"] == "{group}"')
                samples = RecordData.filter_samples(samples, conditions)
                labels = RecordData.extract_labels(samples, label_query)
                result = self.calc_auroc(value_name, scores, labels, record)
            else:
                result = self.calc_aggfunc(value_name, aggfunc, scores, record)
            adhoc.verbose_print(record)
            self.append_score(record)
            adhoc.saved(self.scorepath, "Record of score//スコアの保存先")
        return result

    def get_grouped_scores(self, samples, name, groupby=None):
        group_scores = {}
        for sample in samples:
            model = sample['_model']
            datatag = sample['_dataset']
            score = sample[name]
            key = (model, datatag, '')
            if key not in group_scores:
                group_scores[key] = []
            group_scores[key].append(score)
            if groupby:
                group = sample[groupby]
                key = (model, datatag, str(group))
                if key not in group_scores:
                    group_scores[key] = []
                group_scores[key].append(score)
        return group_scores
    
    def get_value_name(self, datatag, group, name):
        value_name = datatag if group=='' else f'{datatag}[{group}]'
        if name != "exact_match" and name != 'EM':
            value_name = f"{value_name}/{name}"
        return value_name

    def calc_aggfunc(self, value_name, aggfunc, scores, record):
        if 'sum' == aggfunc:
            self.update(record['model'], value_name, sum(scores))
            record['sum'] = sum(scores)
            return record['sum']
        self.update(record['model'], value_name, np.mean(scores))
        record['mean'] = np.mean(scores)
        record_ci95(scores, record)
        return record['mean']
        
    def calc_auroc(self, value_name, scores, labels, record):
        adhoc.safe_import('sklearn', 'scikit-learn')
        from sklearn.metrics import roc_curve, auc
        _labels = []
        _scores = []
        for s, l in zip(scores, labels):
            if not math.isnan(s):
                _labels.append(l)
                _scores.append(s)
        fpr_list, tpr_list, thresholds = roc_curve(_labels, _scores)
        #adhoc.verbose_print(f'閾値 {value_name}', thresholds)
        auroc = auc(fpr_list, tpr_list)
        fpr95 = fpr_list[np.where(tpr_list >= 0.95)[0][0]]
        tpr05 = tpr_list[np.where(fpr_list <= 0.05)[0][-1]]
        self.update(record["model"], f'{value_name}_AUROC', auroc)
        self.update(record["model"], f'{value_name}_FPR95', fpr95)
        self.update(record["model"], f'{value_name}_TPR05', tpr05)
        record.update({
            "AUROC": auroc,
            "FPR95": fpr95,
            "TPR05": tpr05,
        })
        return auroc

    def show(self, transpose=True, width=32):
        if os.path.exists(self.tablepath):
            df = pd.read_csv(self.tablepath)
            # 表示オプションの設定
            pd.set_option("display.width", 512)  # DataFrame全体の幅を設定
            pd.set_option("display.max_colwidth", width)  # 各列の最大幅を設定
            if transpose:
                adhoc.print(df.transpose(), face="")
            else:
                adhoc.print(df, face="")


@adhoc.from_kwargs
def leaderboard_from_kwargs(**kwargs):
    filepath = adhoc.get(kwargs, "leaderboard|leadersboard")

    if filepath is None:
        project_name = adhoc.get(kwargs, "project_name|project")
        filepath = join_name("leaderboard", project_name, ext="csv")
    if not filepath.endswith('.csv'):
        filepath = f"{filepath}.csv"

    output_path = adhoc.get(kwargs, "output_path")
    if output_path:
        filepath = os.path.join(output_path, filepath)
    
    return LeaderBoard(filepath, **kwargs)

def record_ci95(scores: List[float], record: dict, confidence_level = 0.95):
    stats = adhoc.safe_import("scipy.stats", "scipy")
    data = np.array(scores)
    # 標本サイズ
    n = len(data)
    # 標本平均
    mean = np.mean(data)
    # 標本標準偏差
    std_dev = np.std(data, ddof=1)
    # 標準エラー
    se = std_dev / np.sqrt(n)
    # # 信頼水準（95%信頼区間）
    # # 自由度
    df = n - 1
    # # t分布の臨界値
    t_critical = stats.t.ppf((1 + confidence_level) / 2, df)
    # # 信頼区間の計算
    margin_of_error = t_critical * se
    confidence_interval = (mean - margin_of_error, mean + margin_of_error)
    record["mean"] = round(mean, 2)
    record["stderr"] = round(se, 2)
    record["CI95%"] = confidence_interval
