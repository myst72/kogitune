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

    def load_from_map(self, path, kwargs):
        if path.startswith('maxmean_'):
            path = path[8:]
            m = super().load_from_map(path, kwargs)
            return _MaxMeanSim(m, **kwargs)
        return super().load_from_map(path, kwargs)

    def load_default(self, path, kwargs):
        adhoc.print(f"Metric{path}is not found.//評価尺度{path}は見つかりません.", color='red')
        adhoc.print('Select//候補', sorted(METRICS_MAP.keys()), color='magenta', face='')
        return NullMetric(**kwargs)

MetricLoader(METRICS_MAP).register("metric")

class Metric(adhoc.AdhocObject):
    SCHEME = 'metric'
    
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

class _MaxMeanSim(Metric):

    def __init__(self, inner:Metric, **kwargs):
        super().__init__(**kwargs)
        self.inner = inner
        self.name = f"{self.inner.name}_maxmean"
        self.extractor = self.load('extractor', 'textsim_split|=lines', **kwargs)

    def check(self, samples):
        return self.inner.check(samples)

    def calc_s(self, candidate: str, reference: str) -> float:
        # テキストを行ごとに分割
        candidate_lines = self.extractor.extract(candidate)
        reference_lines = self.extractor.extract(reference)

        # 各行の最大類似度を計算
        max_similarities = []
        for candidate_line in candidate_lines:
            line_similarities = [self.inner.calc_s(candidate_line, ref_line) for ref_line in reference_lines]
            max_similarity = max(line_similarities) if line_similarities else 0  # 行ごとの最大値を取得
            max_similarities.append(max_similarity)

        # 最大類似度の平均を計算
        average_similarity = sum(max_similarities) / len(max_similarities) if max_similarities else 0
        return average_similarity

@adhoc.reg('none')
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

    def calc_s(self, candidate: str, reference: str) -> float:
        return 1.0 if reference == candidate else 0.0

ExactMatch.register("exact_match|EM")

