from typing import List, Union
import numpy as np

from ..loads.commons import *


## base class

METRICS_MAP = {}

def split_digit(s):
    d = ""
    while s[-1].isdigit():
        d = s[-1] + d
        s = s[:-1]
    return s, None if d == "" else int(d)


class MetricLoader(adhoc.AdhocLoader):

    def load(self, path: str, tag: str, kwargs):
        path, k = split_digit(path)
        if k is not None:
            kwargs["k"] = k
        scheme = path.lower().replace("-", "_").partition(":")[0]
        if scheme in METRICS_MAP:
            return METRICS_MAP[scheme](path, kwargs)
        adhoc.notice('Available metrics//利用可能な評価尺度', list(METRICS_MAP.keys()))
        raise KeyError(scheme)


MetricLoader().register("metric")


class Metric(adhoc.LoaderObject):
    """
    Base class for evaluators that use a model to obtain answers for generated prompts,
    evaluate them based on specified METRICS_MAP, and calculate scores.
    """

    def __init__(self, name: str, kwargs):
        self.name = name
        self.path = name
        self.scale = 100


    def eval(self, candidate: Union[str, List[str]], reference: Union[str, List[str]]) -> float:
        if isinstance(reference, str):
            if isinstance(candidate, str):
                return self.eval_s(candidate, reference, None)
            reference = [reference] * len(candidate)
        scores = []
        for c, r in zip(candidate, reference):
            scores.append(self.eval_s(c, r, None))
        return np.array(scores).mean()

    def eval_s(self, candidate: str, reference: str, sample=None) -> float:
        if (self.strict and reference == candidate) or reference in candidate:
            return 1.0
        return 0.0

    def extract_pairs(self, sample: dict):
        return sample["output"], sample["reference"],

    def evaluate(self, samplelist: List[dict], force_eval=False):
        scores = []
        for sample in adhoc.tqdm(samplelist, desc=f"{self.name}"):
            if force_eval or self.name not in sample:
                candidates, reference = self.extract_pairs(sample)
                sample[self.name] = self.eval_s(candidates, reference, sample) * self.scale
                self.verbose_print(sample)
            if self.name in sample:
                scores.append(sample[self.name])
        if len(scores) == 0:
            adhoc.notice(f"No score//ひとつも計算できないよ", self.name)
            return None
        return np.array(scores).mean()


    @classmethod
    def register(cls, schemes):
        global METRICS_MAP
        for scheme in schemes.split("|"):
            METRICS_MAP[scheme] = cls


class ExactMatch(Metric):
    """
    コード評価用Evaluatorクラス。
    """

    def __init__(self, path, kwargs):
        Metric.__init__(self, "exact_match", kwargs)
        self.strict = kwargs.get("strict", True)

    def eval_s(self, candidate: str, reference: str, sample:dict=None) -> float:
        if (self.strict and reference == candidate) or reference in candidate:
            return 1.0
        return 0.0


ExactMatch.register("exact_match|EM")




