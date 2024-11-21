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
        if path.endswith('_maxmean'):
            path = path[:-8]
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

    def append_results(self, key:str, value: Any):
        if not hasattr(self, 'additional_results'):
            self.additional_results = {}
        if key not in self.additional_results:
            self.additional_results[key] = []
        self.additional_results[key].append(value)

    def results(self, key:str, value: Any):
        results = {key: value}
        if hasattr(self, 'additional_results'):
            for akey, avalue in self.additional_results.items():
                if akey.startswith('_'):
                    results[f'{key}{akey}'] = avalue
                else:
                    results[f'{akey}'] = avalue
        return results

    def calc(self, candidates:List[str], references:List[str], suffix='')->dict:
        candidates = listfy(candidates)
        references = listfy(references)
        if isinstance(candidates[0], list):
            flat_candidates = []
            flat_references = []
            n = len(candidates[0])
            for candidate, reference in zip(candidates, references):
                flat_candidates.extend(candidate)
                flat_references.extend([reference] * n)
            return self.calc_m(flat_candidates, flat_references, n, suffix)
        if isinstance(references[0], list):
            flat_candidates = []
            flat_references = []
            n = len(references[0])
            for candidate, reference in zip(candidates, references):
                flat_references.extend([candidate] * n)
                flat_candidates.extend(reference)
            return self.calc_m(flat_candidates, flat_references, n, suffix)
        return self.calc_m(candidates, references, 1, suffix)
    
    def calc_m(self, candidates:List[str], references:List[str], n=1, suffix='')->dict:
        scores = []
        for candidate, reference in zip(listfy(candidates), listfy(references)):
            scores.append(self.calc_s(candidate, reference))
        return self.results(f"{self.nametag}{suffix}", 
                            ('mean', self.flatten_mean(scores, n))) 


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
        self.extractor = self.load('extractor', 'textsim_split|=lines', **kwargs)
        self.name = f"{self.inner.name}_{self.extractor.path}"

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

        self.append_results('_sim', list(zip(candidate_lines, max_similarities)))

        # 最大類似度の平均を計算
        mean_similarity = sum(max_similarities) / len(max_similarities) if max_similarities else 0
        return mean_similarity

@adhoc.reg('none')
class NullMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "none"

    def check(self, samples):
        return False

@adhoc.reg("exact_match|em")
class ExactMatch(Metric):
    """
    コード評価用Evaluatorクラス。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "exact_match"

    def calc_s(self, candidate: str, reference: str) -> float:
        return 1.0 if f'{reference}' == candidate else 0.0

