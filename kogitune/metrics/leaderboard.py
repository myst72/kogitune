from typing import List
import os
import json
import numpy as np
import pandas as pd
from ..loads.commons import *


class LeaderBoard(object):
    def __init__(self, filepath="leaderboard.csv", /, **kwargs):
        self.tablepath = filepath
        self.scorepath = filepath.replace('.csv', '_score.jsonl')
        safe_makedirs(self.tablepath)
        safe_makedirs(self.scorepath)
        # if not adhoc.get(kwargs, "overwrite|=True"):
        #     if os.path.exists(self.tablepath):
        #         os.unlink(self.tablepath)
        #     if os.path.exists(self.scorepath):
        #         os.unlink(self.scorepath)

    def update(self, index: str, key: str, value: float):
        found = False
        if os.path.exists(self.tablepath):
            df = pd.read_csv(self.tablepath)
            json_list = df.to_dict(orient="records")
            for d in json_list:
                if d["index"] == index:
                    d[key] = value
                    scores = [v for k, v in d.items()]
                    d["score"] = np.array(scores[2:]).mean()
                    found = True
        else:
            json_list = []
        if found == False:
            json_list.append(
                {
                    "index": index,
                    "score": value,
                    key: value,
                }
            )
        df = pd.DataFrame(json_list).sort_values(by="score", ascending=False)
        df.to_csv(self.tablepath, index=False)

    def append_score(self, record: dict):
        with open(self.scorepath, "a", encoding="utf-8") as w:
            print(json.dumps(record, ensure_ascii=False), file=w)

    def score_testdata(self, testdata, metric_name, **kwargs):
        groupby = kwargs.get('groupby', None)
        group_scores = {"": []}
        for sample in testdata.samples():
            score = sample[metric_name]
            group_scores[""].append(score)
            if groupby:
                group = sample[groupby]
                if group not in group_scores:
                    group_scores[group] = []
                scores = group_scores[group]
                scores.append(score)
        modeltag, datatag = testdata.tags
        for group in group_scores.keys():
            scores = np.array(group_scores[group])
            key = f"{datatag}[{group}]" if group != '' else datatag
            if metric_name != "exact_match":
                key = f"{key}/{metric_name}"
            self.update(modeltag, key, scores.mean())
            record = {
                "model": modeltag,
                "data": datatag,
                "group": group,
                "metric": metric_name,
                "mean": scores.mean(),
                "count": len(scores),
                "scores": list(round(v, 2) for v in scores),
            }
            adhoc.verbose_print(adhoc.dump(record))
            self.append_score(record)
            adhoc.saved(self.scorepath, "Record of score///スコアの保存先")


    def pivot_table(self, samples:List[dict], name:List[str], index=None, groupby=None):
        if index is None:
            index = '{dataset}' if groupby is None else 'ALL'
        if index.startswith('{') and index.endswith('}'):
            index = adhoc.get(sample[0], index[1:-1])
        group_scores = {index: []}
        name = name.split('#')[0]
        tag = name.split('#')[-1]
        for sample in samples:
            score = sample[name]
            group_scores[index].append(score)
            if groupby:
                group = sample[groupby]
                if group not in group_scores:
                    group_scores[group] = []
                scores = group_scores[group]
                scores.append(score)
        for group in group_scores.keys():
            scores = np.array(group_scores[group])
            self.update(group, tag, scores.mean())


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
