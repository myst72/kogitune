from typing import List
import os
import json
import numpy as np
import pandas as pd
from ..loads.commons import *


class LeaderBoard(object):
    def __init__(self, filepath="leaderboard.csv"):
        self.filepath = filepath
        safe_makedirs(filepath)

    def update(self, index: str, key: str, value: float):
        found = False
        if os.path.exists(self.filepath):
            df = pd.read_csv(self.filepath)
            json_list = df.to_dict(orient="records")
            for d in json_list:
                if d["index"] == index:
                    d[key] = value
                    scores = [v for k, v in d.items()]
                    d["score"] = np.array(scores[2:]).mean()
                    found = True
                    return
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
        df.to_csv(self.filepath, index=False)


    def score(self, testdata, metric_name, groupby=None, save_to=None):
        group_scores = {"ALL": []}
        for sample in testdata.samples():
            score = sample[metric_name]
            group_scores["ALL"].append(score)
            if groupby:
                group = sample[groupby]
                if group not in group_scores:
                    group_scores[group] = []
                scores = group_scores[group]
                scores.append(score)
        modeltag, datatag = testdata.tags
        for group in group_scores.keys():
            scores = np.array(group_scores[group])
            testname = (
                datatag if metric_name == "exact_match" else f"{datatag}/{metric_name}"
            )
            self.update(modeltag, testname, scores.mean())
            if save_to:
                if not isinstance(save_to, str):
                    save_to = f"{datatag}_score.jsonl"
                record = {
                    "model": modeltag,
                    "data": datatag,
                    "group": group,
                    "metric": metric_name,
                    "mean": scores.mean(),
                    "scores": list(round(v, 2) for v in scores),
                }
                save_score(save_to, record)
                adhoc.saved(save_to, "スコアの記録")

    def show(self, transpose=True, width=32):
        if os.path.exists(self.filepath):
            df = pd.read_csv(self.filepath)
            # 表示オプションの設定
            pd.set_option("display.width", 512)  # DataFrame全体の幅を設定
            pd.set_option("display.max_colwidth", width)  # 各列の最大幅を設定
            if transpose:
                adhoc.print(df.transpose(), face="")
            else:
                adhoc.print(df, face="")


def save_score(score_file, result: dict):
    safe_makedirs(score_file)
    with open(score_file, "a", encoding="utf-8") as w:
        print(json.dumps(result, ensure_ascii=False), file=w)

@adhoc.from_kwargs
def leaderboard_from_kwargs(**kwargs):
    filepath = adhoc.get(kwargs, "leaderboard|leadersboard")
    if filepath is None:
        project_name = adhoc.get(kwargs, "project_name|project")
        filepath = join_name("leaderboard", project_name, ext="csv")
    elif not filepath.endswith('.csv'):
        filepath = f"{filepath}.csv"
    return LeaderBoard(filepath)

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
