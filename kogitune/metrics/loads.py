from typing import List, Union

from ..loads.commons import *
import numpy as np

## base class

METRICS_MAP = {}

class MetricLoader(adhoc.AdhocLoader):

    def add_kwargs(self, path: str, kwargs):
        path = self.parse_k_from_path(path, kwargs)
        return path, kwargs

    def load_default(self, path, kwargs):
        adhoc.print(f"Metric{path}is not found.///評価尺度{path}は見つかりません.", color='red')
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
        # print('@m', len(candidates), candidates)
        # print('@m', len(references), references)
        for candidate, reference in zip(listfy(candidates), listfy(references), strict=True):
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

## base class

TASK_MAP = {}

class TaskLoader(adhoc.AdhocLoader):
    pass

TaskLoader(TASK_MAP).register("task")

def fmt(template:str, sample:dict):
    return template.format(**sample)


class Task(adhoc.AdhocObject):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.template = None
        self.chat_mode = True
        self.verbose = VerboseCounter(**kwargs)
        self.progress_bar = adhoc.progress_bar()
        self.heading_messages = adhoc.get(kwargs, 'heading_messages')
        self.init_kwargs = {**kwargs}

    @property
    def tasktag(self):
        name = self.name if hasattr(self, "name") else self.path
        return self.tag if self.tag != '' else name

    def start_progress_bar(self, total:int, desc:str=None):
        self.progress_bar = adhoc.progress_bar(total=total, desc=desc)

    def end_progress_bar(self):
        if self.progress_bar:
            self.progress_bar.close()
            self.progress_bar = adhoc.progress_bar()

    def prepare(self, samples: List[dict]):
        template = None
        for sample in samples:
            if template is None:
                if self.template is not None:
                    template = self.template
                else:
                    template = self.guess_template(sample)
                    if template:
                        adhoc.verbose_print('[Inferred Template]', dump=template, once=True,
                                            if_dislike={'template_config': repr("(template.json)")})
                        self.template = template
                        if self.shots > 0 and 'shot_messages' in template:
                            self.heading_messages = template['shot_messages']
                            self.set_few_shots()

            if template:
                try:
                    self.apply_template(sample, template)
                except KeyError as e:
                    adhoc.print("テンプレートがデータにマッチしません。")
                    adhoc.print("テンプレート", adhoc.dump(template), face='')
                    adhoc.print("データ", adhoc.dump(sample), face='')
                    adhoc.exit(throw=e)
            else:
                self.transform(sample)

    def guess_template(self, sample):
        raise NotImplementedError()
    
    def set_few_shots(self):
        if self.heading_messages and self.shots > 0:
            if self.shots *2 != len(self.heading_messages):
                self.heading_messages = self.heading_messages[:self.shots*2]
                shots = len(self.heading_messages)//2
                if self.shots == shots:
                    return
                # 再調整
                self.shots = shots
                self.name = f'{self.shots}-shot'
                adhoc.verbose_print(f'{self.shots}-shot//ショット', dump=self.heading_messages)


    def apply_template(self, sample:dict, template:dict):
        pass

    def format(self, template, key, sample):
        template_format = template[key]
        return template_format.format(**sample)

    def transform(self, sample):
        adhoc.print(f"{self.name}タスク用のテンプレートがありません。")
        adhoc.print(adhoc.dump(sample), face='')
        adhoc.exit(throw=ValueError(f"{self.name}タスク用のテンプレートがありません。"))


    def eval(self, model, samples: List[dict]):
        pass

    def calc(self, metric: Metric, samples: List[dict]):
        candidates = self.extract_values(samples, "_output")
        references = self.extract_values(samples, "_reference")
        results = metric.calc(candidates, references)
        self.update_values(samples, results)
        return results

    @property
    def default_metrics():
        return []

    def extract_values(self, samples: List[dict], key:str):
        return [sample[key] for sample in samples]

    def update_kwargs(self, samples:List[dict], /, **kwargs):
        items = list(kwargs.items())
        for sample in samples:
            for k, v in items:
                sample[k] = v

    def update_values(self, samples: List[dict], results: dict):
        """
        results = {"output": scores}
        """
        for key, outputs in results.items():
            if isinstance(outputs, tuple):
                # ('mean', scores) 形式への対応
                outputs = outputs[1]
            assert len(outputs) >= len(samples), f"{len(outputs)} == {len(samples)}: {outputs}"
            for i, sample in enumerate(samples):
                sample[key] = outputs[i]

    def verbose_samples(self, samples: List[dict]):
        with self.verbose as verbose:
            for sample in samples:
                verbose.print_sample(sample)

    @classmethod
    def register(cls, schemes):
        global TASK_MAP
        for scheme in schemes.split("|"):
            TASK_MAP[scheme] = cls

