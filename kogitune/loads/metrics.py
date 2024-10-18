from .commons import *
import numpy as np

## base class

METRICS_MAP = {}

class MetricLoader(adhoc.AdhocLoader):

    def load_modules(self, path, kwargs):
        from .metrics_textsim import EditSim
        from .metrics_python import PassAtK

    def add_kwargs(self, path: str, kwargs):
        path = self.parse_k_from_path(path, kwargs)
        return path, kwargs

    def load_default(self, path, kwargs):
        adhoc.print(f"Metric{path}is not found.//評価尺度{path}は見つかりません.", color='red')
        adhoc.print('Select//候補', sorted(METRICS_MAP.keys()), color='magenta', face='')
        return NullMetric(**kwargs)

MetricLoader(METRICS_MAP).register("metric")

class Metric(adhoc.AdhocObject):
    """
    Base class for evaluators that use a model to obtain answers for generated prompts,
    evaluate them based on specified METRICS_MAP, and calculate scores.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scale = kwargs.get('scale', 100)

    @property
    def nametag(self):
        return self.tag if self.tag != '' else self.name

    def check(self, samples):
        return True

    def calc(self, candidates:List[str], references:List[str], suffix='')->dict:
        candidates = listfy(candidates)
        references = listfy(references)
        if len(candidates) > 0 and isinstance(candidates[0], list):
            n = len(candidates[0])
            flat_candidates = []
            flat_references = []
            for candidate, reference in zip(candidates, references):
                flat_candidates.extend(candidate)
                flat_references.extend([reference] * n)
            return self.calc_m(flat_candidates, flat_references, n, suffix)
        return self.calc_m(candidates, references, 1, suffix)
    
    def calc_m(self, candidates:List[str], references:List[str], n=1, suffix='')->dict:
        scores = []
        for candidate, reference in zip(listfy(candidates), listfy(references)):
            scores.append(self.calc_s(candidate, reference))
        return {f"{self.nametag}{suffix}": ('mean', self.flatten_mean(scores, n))}

    def flatten_mean(self, scores:List[float], n):
        if n == 1:
            return scores
        else:
            ss = []
            mean_scores = []
            for score in scores:
                ss.append(score)
                if len(ss) == n:
                    mean_scores.append(np.mean(ss))
                    ss=[]
            
            assert len(scores) == len(mean_scores) * n
            return mean_scores

    def calc_s(self, candidate:str, reference:str) -> float:
        return 0
    
    @classmethod
    def register(cls, schemes):
        global METRICS_MAP
        for scheme in schemes.split("|"):
            METRICS_MAP[scheme] = cls
            METRICS_MAP[scheme.lower().replace('_', '')] = cls

class NullMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "none"

    def check(self, samples):
        return False


class ExactMatch(Metric):
    """
    コード評価用Evaluatorクラス。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "exact_match"
        self.strict = kwargs.get("strict", True)

    def calc_s(self, candidate: str, reference: str) -> float:
        if (self.strict and reference == candidate) or reference in candidate:
            return 1.0
        return 0.0

ExactMatch.register("exact_match|EM")

