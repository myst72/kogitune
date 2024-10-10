from typing import List
import os
import json
import numpy as np
import pandas as pd
from ..loads.commons import *

## Record

class RecordLoader(adhoc.AdhocLoader):

    def load_from_map(self, path, kwargs):
        if path.endswith('.txt'):
            df = pd.read_table(path, sep='\n', header=None)
            df.columns = ['text']
            dict_rows = df.to_dict(orient='records')
            return RecordData(path, dict_rows, **kwargs)
        if path.endswith('.csv'):
            df = pd.read_csv(path)
            df.columns = [str(name).lower().replace(" ", "_") for name in df.columns]
            dict_rows = df.to_dict(orient='records')
            return RecordData(path, dict_rows, **kwargs)
        if path.endswith('.tsv'):
            df = pd.read_csv(path, sep='\t')
            df.columns = [str(name).lower().replace(" ", "_") for name in df.columns]
            dict_rows = df.to_dict(orient='records')
            return RecordData(path, dict_rows, **kwargs)
        if path.endswith('.xlsx'):
            df = pd.read_excel(path)
            df.columns = [str(name).lower().replace(" ", "_") for name in df.columns]
            dict_rows = df.to_dict(orient='records')
            return RecordData(path, dict_rows, **kwargs)
        if path.endswith('.xls'):
            df = pd.read_excel(path)
            df.columns = [str(name).lower().replace(" ", "_") for name in df.columns]
            dict_rows = df.to_dict(orient='records')
            return RecordData(path, dict_rows, **kwargs)
        df = pd.read_json(path, lines=True)
        dict_rows = df.to_dict(orient='records')
        return RecordData(path, dict_rows, **kwargs)

RecordLoader({}).register("record")


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
            adhoc.verbose_print(f'既存の{self.save_path}を使います。嫌なら`resume=False`してね')            
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
                    # scores = [v for k, v in d.items()]
                    # d["score"] = np.array(scores[2:]).mean()
                    found = True
        else:
            json_list = []
        if found == False:
            json_list.append(
                {
                    "index": index,
                    # "score": value,
                    key: value,
                }
            )
        # df = pd.DataFrame(json_list).sort_values(by="score", ascending=False)
        df = pd.DataFrame(json_list)
        df.to_csv(self.tablepath, index=False)

    def append_score(self, record: dict):
        with open(self.scorepath, "a", encoding="utf-8") as w:
            print(json.dumps(record, ensure_ascii=False), file=w)

    def pivot_table(self, samples:dict, name:str, aggfunc = 'mean', /, **kwargs):
        if isinstance(name, dict):
            results = name
            for name in results.keys():
                aggfunc = None
                if isinstance(results[name], str):
                    aggfunc = results[name]
                elif isinstance(results[name], tuple):
                    aggfunc = results[name][0]  # ('mean', scores) 形式
                if aggfunc:
                    self.pivot_table(samples, name, aggfunc, **kwargs)
            return
        
        groupby = kwargs.get('groupby', kwargs.get('groupby', None))
        grouped_scores = self.get_grouped_scores(samples, name, groupby)

        label_key = None
        if aggfunc.startswith('AUC:') or aggfunc.startswith('ROC:'):
            label_key = aggfunc[4:]

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
                "mean": np.mean(scores),
            }
            if label_key:
                labels = self.get_values(samples, label_key, model, datatag, group, groupby)
                self.calc_auroc(value_name, scores, labels, record)
            else:
                self.calc_aggfunc(aggfunc, scores, value_name, record)
            adhoc.verbose_print(record)
            self.append_score(record)
            adhoc.saved(self.scorepath, "Record of score//スコアの保存先")

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

    def calc_aggfunc(self, aggfunc, scores, value_name, record):
        if 'mean' in aggfunc:
            self.update(record['model'], value_name, np.mean(scores))
        if 'sum' in aggfunc:
            self.update(record['model'], value_name, sum(scores))

    def calc_mean(self, value_name, scores, record):
        self.update(record['model'], value_name, np.mean(scores))

    def calc_auroc(self, value_name, scores, labels, record):
        from sklearn.metrics import roc_curve, auc
        fpr_list, tpr_list, thresholds = roc_curve(labels, scores)
        adhoc.verbose_print(f'閾値 {value_name}', thresholds)
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

# def calc_mean(scores: List[float], record: dict):
#     data = np.array(scores)
#     # 標本サイズ
#     n = len(data)
#     # 標本平均
#     mean = np.mean(data)
#     # 標本標準偏差
#     std_dev = np.std(data, ddof=1)
#     # 標準エラー
#     se = std_dev / np.sqrt(n)
#     # # 信頼水準（95%信頼区間）
#     # confidence_level = 0.95
#     # # 自由度
#     # df = n - 1
#     # # t分布の臨界値
#     # t_critical = stats.t.ppf((1 + confidence_level) / 2, df)
#     # # 信頼区間の計算
#     # margin_of_error = t_critical * se
#     # confidence_interval = (mean - margin_of_error, mean + margin_of_error)
#     record["mean"] = round(mean, 2)
#     record["stderr"] = round(se, 2)
#     # results["CI95%"] = confidence_interval
